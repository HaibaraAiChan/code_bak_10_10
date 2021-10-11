import multiprocessing
from multiprocessing import Pool
def worker(name, que):
    que.put("%d is done" % name)


def test():
    pool = multiprocessing.Pool(processes=3)
    m = multiprocessing.Manager()
    q = m.Queue()
    
    workers = pool.apply_async(worker, (33, q))
    pool.close()
    pool.join()
    i=0
    while not q.empty():
        cur = q.get()
        i=i+1
        print(i)
        print(cur)

if __name__ == '__main__':

    test()
    