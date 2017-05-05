from network import Network
import queue


# TODO Make all the following stuff (except last function) clear, and move it to a brand new file for percolation !
class PercolationStuff(Network):
    def __init__(self, base_power=0):
        Network.__init__(self, base_power)

    def floyd_warshall(self):
        fw = [[1000 * self.size] * self.size for _ in range(self.size)]
        for i in range(self.size):
            for v in self.graph[i]:
                fw[i][v] = 1

        for c in range(self.size):
            for a in range(self.size):
                for b in range(self.size):
                    fw[a][b] = min(fw[a][b], fw[a][c] + fw[c][b])

        return fw

    def count_paths(self):
        nbP = [[[0] * self.size for _ in range(self.size)] for _ in range(self.size + 1)]
        for i in range(self.size):
            for v in self.graph[i]:
                nbP[0][i][v] = 1

        for i in range(2, self.size + 1):
            for a in range(self.size):
                for b in range(self.size):
                    for c in range(self.size):
                        nbP[i][a][b] += nbP[1][a][c] * nbP[i - 1][c][b]

        return nbP

    def shortest_paths(self):
        SP = [[(-1, 0)] * self.size for _ in range(self.size)]

        for i in range(self.size):
            fl = queue.Queue()
            fl.put((i, i, 0))
            while not fl.empty():
                cur = fl.get()
                if SP[i][cur[0]][0] == -1:
                    for v in self.graph[cur[0]]:
                        fl.put((v, cur[0], cur[2] + 1))
                SP[i][cur[0]] = (cur[2], SP[i][cur[0]][1] + SP[i][cur[1]][1])

        return SP

    def compute_percolation(self):
        """
        Computes the percolation of the graph of the network.
        :return: the percolation
        """
        perc = [0] * self.size
        visited = [0] * self.size
        prov = [0] * self.size
        count = 1
        for i in range(self.size):
            for j in range(self.size):
                prov[j] = -1

            # We compute with a bfs the last node prov[j]
            # on a shortest path between i and j
            queue = [(i, -1)]
            while len(queue) > 0:
                cur, last = queue.pop()
                if visited[cur] < count:
                    visited[cur] = count
                    prov[cur] = last
                    for v in self.graph[cur]:
                        queue.append((v, cur))

            # For each node j, we follow the shortest path from
            # j to i using the array prov computed previously
            for j in range(self.size):
                if i != j and visited[j] == count:
                    cur = j
                    while cur != -1:
                        perc[cur] += 1
                        cur = prov[cur]
            count += 1
        return perc

    def compute_percolation_betweenness(self):
        SP = self.shortest_paths()

        B = []

        for n in range(self.size):
            p = 0
            for m in range(self.size):
                for o in range(self.size):
                    if SP[m][n][0] + SP[n][o][0] == SP[m][o][0]:
                        p += SP[m][n][1] * SP[n][o][1] / SP[m][o][1]
            B.append(p / (self.size - 1) / (self.size - 2))

        return B

    def compute_percolation_centrality(self, I):
        SP = self.shortest_paths()

        P = []

        sI = sum(I)

        for n in range(self.size):
            p = 0
            for m in range(self.size):
                for o in range(self.size):
                    if SP[m][n][0] + SP[n][o][0] == SP[m][o][0]:
                        w = I[m] / (sI - I[n])
                        p += SP[m][n][1] * SP[n][o][1] / SP[m][o][1] * w
            P.append(p / (self.size - 2))

        return P

    # Legacy
    #def compute_percolation_betweenness_slow(self):
    #    fw = self.floyd_warshall()
    #    nbP = self.count_paths()

    #    B = []

    #    for g in range(self.size):
    #        p = 0
    #        for s in range(self.size):
    #            for t in range(self.size):
    #                if fw[s][g] + fw[g][t] == fw[s][t]:
    #                    p += nbP[fw[s][g]][s][g] * nbP[fw[g][t]][g][t] / nbP[fw[s][t]][s][t]
    #        B.append(p / (self.size - 1) / (self.size - 2))

    #    return B

    #def compute_percolation_centrality_slow(self, I):
    #    fw = self.floyd_warshall()
    #    nbP = self.count_paths()

    #    P = []

    #    sI = sum(I)

    #    for g in range(self.size):
    #        p = 0
    #        for s in range(self.size):
    #            for t in range(self.size):
    #                if fw[s][g] + fw[g][t] == fw[s][t]:
    #                    w = I[s] / (sI - I[g])
    #                    p += nbP[fw[s][g]][s][g] * nbP[fw[g][t]][g][t] / nbP[fw[s][t]][s][t] * w
    #        P.append(p / (self.size - 2))

    #    return P
