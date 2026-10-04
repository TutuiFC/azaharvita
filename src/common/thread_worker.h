// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <queue>

#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "common/unique_function.h"
#ifdef __PSVITA__
#include "common/vita_diag.h"
#endif

namespace Common {

template <class StateType = void>
class StatefulThreadWorker {
    static constexpr bool with_state = !std::is_same_v<StateType, void>;

    struct DummyCallable {
        int operator()(std::size_t) const noexcept {
            return 0;
        }
    };

    using Task =
        std::conditional_t<with_state, UniqueFunction<void, StateType*>, UniqueFunction<void>>;
    using StateMaker =
        std::conditional_t<with_state, std::function<StateType(std::size_t)>, DummyCallable>;

public:
    /**
     * @param pin_to_cores Solo tiene efecto en PS Vita. Ata cada hilo a un
     *        nucleo de usuario distinto (en round-robin entre los tres que hay).
     *        Es opcional y por defecto no se hace porque no todos los pools lo
     *        quieren: el del rasterizador reparte trabajo de calculo y le
     *        interesa ocupar nucleos distintos de verdad, pero el de ficheros
     *        pasa la vida bloqueado esperando a la tarjeta y atarlo no aporta
     *        nada. Ver Common::VitaPinThreadToUserCore.
     * @param core_offset Primer nucleo del reparto. El hilo de emulacion esta
     *        atado al nucleo 0 (ver main.cpp), asi que un pool cuyo dueno
     *        rasterice ademas en su propio hilo tiene que empezar en el 1: si
     *        no, uno de los hilos del pool comparte nucleo con quien le da
     *        trabajo. Ver el reparto por bandas del rasterizador.
     */
    explicit StatefulThreadWorker(std::size_t num_workers, std::string_view name,
                                  StateMaker func = {}, bool pin_to_cores = false,
                                  unsigned int core_offset = 0)
        : workers_queued{num_workers}, thread_name{name} {
        const auto lambda = [this, func, pin_to_cores, core_offset](std::stop_token stop_token,
                                                                    std::size_t index) {
            Common::SetCurrentThreadName(thread_name.data());
#ifdef __PSVITA__
            // El FPSCR es por hilo y se guarda en cada cambio de contexto, asi
            // que activarlo en main() no vale para estos. Y son justo los hilos
            // donde importa: los que sombrean pixeles.
            Common::VitaEnableFastFloatMode();
            if (pin_to_cores) {
                Common::VitaPinThreadToUserCore(
                    static_cast<unsigned int>(index) + core_offset, thread_name.data());
                // Por encima del compilador de shaders: ver kVitaPriorityHelper.
                Common::VitaSetThreadPriority(Common::kVitaPriorityHelper, thread_name.data());
            }
#else
            (void)pin_to_cores;
            (void)core_offset;
#endif
            {
                [[maybe_unused]] std::conditional_t<with_state, StateType, int> state{func(index)};
                while (!stop_token.stop_requested()) {
                    Task task;
                    {
                        std::unique_lock lock{queue_mutex};
                        // Solo se avisa si hay alguien esperando de verdad en la
                        // barrera. Antes se hacia SIEMPRE que la cola quedaba
                        // vacia, y eso es una llamada al planificador por vuelta
                        // del bucle y por hilo -- miles por fotograma en el
                        // rasterizador -- que ademas despierta a los demas hilos
                        // del pool para nada. Con la espera activa de
                        // WaitForRequestsSpin el caso normal es que nadie este
                        // durmiendo, asi que este aviso sobra del todo.
                        //
                        // 'waiters' se toca siempre con queue_mutex cogido (aqui
                        // y en WaitForRequests), asi que no hace falta que sea
                        // atomico ni hay carrera posible: quien se va a dormir
                        // lo incrementa antes de soltar el cerrojo.
                        if (requests.empty() && waiters != 0) {
                            wait_condition.notify_all();
                        }
                        Common::CondvarWait(condition, lock, stop_token,
                                            [this] { return !requests.empty(); });
                        if (stop_token.stop_requested()) {
                            break;
                        }
                        task = std::move(requests.front());
                        requests.pop();
                    }
                    if constexpr (with_state) {
                        task(&state);
                    } else {
                        task();
                    }
                    ++work_done;
                }
            }
            ++workers_stopped;
            wait_condition.notify_all();
        };
        threads.reserve(num_workers);
        for (std::size_t i = 0; i < num_workers; ++i) {
            threads.emplace_back(lambda, i);
        }
    }

    StatefulThreadWorker& operator=(const StatefulThreadWorker&) = delete;
    StatefulThreadWorker(const StatefulThreadWorker&) = delete;

    StatefulThreadWorker& operator=(StatefulThreadWorker&&) = delete;
    StatefulThreadWorker(StatefulThreadWorker&&) = delete;

    void QueueWork(Task work) {
        {
            std::unique_lock lock{queue_mutex};
            requests.emplace(std::move(work));
            ++work_scheduled;
        }
        condition.notify_one();
    }

    void WaitForRequests(std::stop_token stop_token = {}) {
        std::stop_callback callback(stop_token, [this] {
            for (auto& thread : threads) {
                thread.request_stop();
            }
        });
        std::unique_lock lock{queue_mutex};
        ++waiters;
        wait_condition.wait(lock, [this] {
            return workers_stopped >= workers_queued || work_done >= work_scheduled;
        });
        --waiters;
    }

    /**
     * Barrera con espera ACTIVA acotada antes de pasar a dormir.
     *
     * El rasterizador llega aqui una vez por triangulo -- unos mil por
     * fotograma -- y cada paso por la barrera normal cuesta, como minimo, un
     * cerrojo, una variable de condicion y dos viajes al planificador: dormir a
     * este hilo y volver a despertarlo cuando el ultimo trabajador termina. Esa
     * latencia esta en el camino critico mil veces por fotograma.
     *
     * Girando en vacio unas vueltas se sale sin tocar nada de eso en el caso
     * normal, que es que los trabajadores acaben casi a la vez que este hilo:
     * quien llama rasteriza su propia banda antes de esperar, asi que cuando
     * llega aqui a los demas les queda muy poco.
     *
     * La espera activa NO roba tiempo de calculo: el hilo de emulacion esta
     * atado al nucleo 0 y los del pool a los nucleos 1 y 2 (ver core_offset).
     * Si se agotan las vueltas -- porque una banda salio mucho mas cara que las
     * otras, o porque el sistema le quito el nucleo a un trabajador -- se cae a
     * la barrera de siempre y se duerme, que es lo correcto para esperas largas.
     *
     * 'yield' es una pista para el procesador, no un cambio de contexto: en
     * Cortex-A9 no hace nada salvo dejar respirar a la tuberia.
     */
    void WaitForRequestsSpin(u32 spin_rounds) {
        for (u32 i = 0; i < spin_rounds; i++) {
            // Adquisicion: el incremento de work_done va DESPUES de ejecutar la
            // tarea, asi que ver el contador alto garantiza ver tambien todo lo
            // que la tarea escribio (los pixeles de su banda).
            if (work_done.load(std::memory_order_acquire) >=
                work_scheduled.load(std::memory_order_relaxed)) {
                return;
            }
#if defined(__ARM_ARCH)
            __asm__ __volatile__("yield" ::: "memory");
#endif
        }
        WaitForRequests();
    }

    const std::size_t NumWorkers() const noexcept {
        return threads.size();
    }

private:
    std::queue<Task> requests;
    std::mutex queue_mutex;
    std::condition_variable_any condition;
    std::condition_variable wait_condition;
    std::atomic<std::size_t> work_scheduled{};
    std::atomic<std::size_t> work_done{};
    std::atomic<std::size_t> workers_stopped{};
    std::atomic<std::size_t> workers_queued{};
    /// Hilos dormidos en la barrera. Siempre bajo queue_mutex. Ver el aviso
    /// condicionado del bucle de trabajo.
    std::size_t waiters{};
    std::string_view thread_name;
    std::vector<std::jthread> threads;
};

using ThreadWorker = StatefulThreadWorker<>;

} // namespace Common
