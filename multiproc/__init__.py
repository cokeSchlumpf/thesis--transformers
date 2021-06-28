import multiprocessing as mp


class Test:

    def my_func(self, x):
        return x ** x

    def main(self):

        pool = mp.Pool(mp.cpu_count())
        result = pool.map(self.my_func, [4, 2, 3])

        print(result)


if __name__ == "__main__":
    Test().main()
