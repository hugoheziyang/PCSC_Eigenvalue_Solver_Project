#include <iostream>





int add(int x, int y) {
    if (x == 0){
        throw std::invalid_argument("x cannot be zero"); 
    }
    return x + y;
}


int main() {
    
    
    return 0;
}