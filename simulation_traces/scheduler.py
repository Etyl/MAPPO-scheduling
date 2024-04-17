
        
class Scheduler:

    def assign(self, request: int, currentState: State) -> list[float]:
        n_nodes = len(currentState.nodeLoadCPU)
        result = [0]*(n_nodes+1)
        for i in range(n_nodes):
            if (currentState.nodeLoadCPU[i] < 0.8 and 
                currentState.nodeLoadIO[i] < 0.8 and 
                currentState.nodeLoadStorage[i] < 0.8):
                result[i] = 1
        if sum(result) == 0:
            result[n_nodes] = 1
        return result


class ML_Scheduler:

    def assign(self, request: int, currentState: State) -> list[float]:
        pass
        # return [0, 0, 0, 0, 1]
