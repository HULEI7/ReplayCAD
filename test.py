class Solution:
    def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
        if not matrix:
            return 0
        rows = len(matrix)
        cols = len(matrix[0])
        # memo[i][j] stores the length of the longest increasing path starting from (i, j)
        memo = [[0] * cols for _ in range(rows)]
    
        # Directions for moving up, down, left, right
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        def dfs(r, c):
            # If the result for this cell is already computed, return it
            if memo[r][c] != 0:
                return memo[r][c]
            # The current cell itself is a path of length 1
            max_len = 1
            # Explore all four directions
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                # Check if the next cell is within bounds and has a greater value
                if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] > matrix[r][c]:
                    # Recursively find the longest path from the next cell
                    path_len = dfs(nr, nc)
                    # Update the maximum length from the current cell
                    max_len = max(max_len, 1 + path_len)
            # Store the computed result in the memoization table
            memo[r][c] = max_len
            return max_len

        overall_max_len = 0
        # Perform DFS from each cell and find the overall maximum length
        for i in range(rows):
            for j in range(cols):
                overall_max_len = max(overall_max_len, dfs(i, j))

        return overall_max_len

# 示例测试
matrix = [
  [9,9,4],
  [6,6,8],
  [2,1,1]
]
sol = Solution()
print(sol.longestIncreasingPath(matrix)) # Output: 4