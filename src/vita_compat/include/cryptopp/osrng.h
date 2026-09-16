// Shim de <cryptopp/osrng.h> para PS Vita.
//
// cryptopp solo declara CryptoPP::AutoSeededRandomPool cuando detecta una
// fuente de entropia del sistema (OS_RNG_AVAILABLE), y solo reconoce la
// CryptoAPI de Windows y /dev/urandom. La Vita no tiene ninguna de las dos, y
// el generador del kernel tampoco vale: un homebrew firmado como "safe" no
// tiene acceso garantizado a esa API. Se usa el de vita_random.h.
//
// Este fichero se antepone al de cryptopp en la ruta de busqueda: incluye el
// original con #include_next y, si el tipo no ha quedado declarado, lo aporta
// sobre sceKernelGetRandomNumber. Asi el codigo de Azahar que usa
// AutoSeededRandomPool (cfg, am, nfc) compila sin tocarlo.

#pragma once

#include_next <cryptopp/osrng.h>

#if !defined(OS_RNG_AVAILABLE) || defined(NO_OS_DEPENDENCE)

#include "vita_random.h"
#include <cryptopp/cryptlib.h>

namespace CryptoPP {

class AutoSeededRandomPool : public RandomNumberGenerator {
public:
    // Misma firma que la de cryptopp; aqui no hay nada que sembrar porque la
    // entropia la aporta el kernel en cada llamada.
    explicit AutoSeededRandomPool(bool blocking = false, unsigned int seedSize = 32) {
        (void)blocking;
        (void)seedSize;
    }

    void Reseed(bool blocking = false, unsigned int seedSize = 32) {
        (void)blocking;
        (void)seedSize;
    }

    void IncorporateEntropy(const byte* input, std::size_t length) override {
        // El RNG del kernel no admite semilla externa.
        (void)input;
        (void)length;
    }

    bool CanIncorporateEntropy() const override {
        return false;
    }

    void GenerateBlock(byte* output, std::size_t size) override {
        VitaRngFill(reinterpret_cast<unsigned char*>(output), size);
    }
};

} // namespace CryptoPP

#endif // !OS_RNG_AVAILABLE
