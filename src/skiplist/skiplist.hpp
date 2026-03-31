#ifndef SKIPLIST_HPP
#define SKIPLIST_HPP

#include <vector>
#include <optional>
#include <functional>

/**
 * @brief Interfaz para una Cola de Prioridad Relajada.
 * @tparam T Tipo de dato (en Viltrum será ExtendedRegion).
 * @tparam Compare Comparador (equivalente a heap_ordering).
 */
template <typename T, typename Compare = std::less<T>>
class Skiplist {
public:
    // Alias para compatibilidad con el benchmark
    using value_type = T;

    /**
     * @brief Constructor que define el grado de relajación.
     * @param num_shards Número de sub-colas internas para reducir contención.
     * @param comp Instancia del comparador de prioridad.
     */
    explicit Skiplist(std::size_t num_shards = 8, Compare comp = Compare());

    /**
     * @brief Inserta un elemento de forma concurrente.
     */
    void push(const T& value);

    /**
     * @brief Intenta extraer el elemento de mayor prioridad.
     * @return std::optional con el valor, o nullopt si la estructura está saturada o vacía.
     */
    std::optional<T> try_pop();

    /**
     * @brief Comprueba si todas las colas internas están vacías.
     */
    bool empty() const;

    /**
     * @brief Retorna el número total de elementos (aproximado o exacto).
     */
    std::size_t size() const;

private:
    // Aquí implementarás la lógica de Multi-Queue o K-LSM.
};

#endif