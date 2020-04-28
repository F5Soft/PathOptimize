import modules.algorithm as alg

if __name__ == "__main__":
    alg.set_trucks()
    alg.set_network()
    alg.network.coverage_allocate()
    for t in alg.network.trucks:
        print(t)
        print(t.coverage)
