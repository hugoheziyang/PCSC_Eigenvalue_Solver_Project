#include <stdexcept>

int add(int x, int y) {
    if (x == 0){
        throw std::invalid_argument("x cannot be zero"); 
    }
    return x + y;
}