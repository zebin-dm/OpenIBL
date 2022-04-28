# from ibl.datasets.pitts import Pittsburgh

# data_path = "/data/zebin/data/Pittsburgh/pitts"
# Pittsburgh(root=data_path)
class Solution:
    def myPow(self, x: float, n: int) -> float:

        if n == 0:
            return 1
        
        if n == 1:
            return x
        
        if n == -1:
            return 1/x
        
        if n > 0:
            abs_n = n
        else:
            abs_n = -n

        def binary_div(n):
            output = list()
            val = n
            while val != 1:
                val1 = val // 2
                val2 = val - val1
                output.append([val2, val1])
                val = val1
            return output

        out_list = binary_div(abs_n)
        print(out_list)
        num = len(out_list)
        out_val = [1, 1]
        for idx in range(num - 1, -1, -1):
            if idx == (num - 1):
                out_val[1] = x
            else:
                out_val[1] = out_val[0] * out_val[1]
            val = out_list[idx]
            if val[0] > val[1]:
                out_val[0] = out_val[1] * x
            else:
                out_val[0] = out_val[1]
        last_val = out_val[0] * out_val[1]
        if abs_n != n:
            last_val = 1.0 / last_val
        return last_val


if __name__ == "__main__":
    x = 2.0
    n = -2
    val = Solution().myPow(x, n)
    print(val)
