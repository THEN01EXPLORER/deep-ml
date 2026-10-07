def dp_drills(weights, values, capacity, m, n, s1, s2):
    # -------------------------
    # 1. 0/1 Knapsack
    # -------------------------
    # n is the number of items according to the problem statement.
    items = min(n, len(weights), len(values))

    dp = [[0] * (capacity + 1) for _ in range(items + 1)]

    for i in range(1, items + 1):
        weight = weights[i - 1]
        value = values[i - 1]

        for w in range(capacity + 1):

            # Don't take the item
            dp[i][w] = dp[i - 1][w]

            # Take the item if it fits
            if weight <= w:
                dp[i][w] = max(
                    dp[i][w],
                    value + dp[i - 1][w - weight]
                )

    knapsack = dp[items][capacity]

    # -------------------------
    # 2. Grid Paths
    # -------------------------
    grid = [[0] * n for _ in range(m)]

    # First row
    for j in range(n):
        grid[0][j] = 1

    # First column
    for i in range(m):
        grid[i][0] = 1

    # Fill remaining cells
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] = grid[i - 1][j] + grid[i][j - 1]

    grid_paths = grid[m - 1][n - 1]

    # -------------------------
    # 3. LCS
    # -------------------------
    len1 = len(s1)
    len2 = len(s2)

    lcs_dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):

            if s1[i - 1] == s2[j - 1]:
                lcs_dp[i][j] = lcs_dp[i - 1][j - 1] + 1

            else:
                lcs_dp[i][j] = max(
                    lcs_dp[i - 1][j],
                    lcs_dp[i][j - 1]
                )

    lcs = lcs_dp[len1][len2]

    # -------------------------
    # Return result
    # -------------------------
    return {
        'knapsack': knapsack,
        'grid_paths': grid_paths,
        'lcs': lcs,
        'complexity': {
            'knapsack': 'O(n*W) time, O(n*W) space',
            'grid_paths': 'O(m*n) time, O(m*n) space',
            'lcs': 'O(m*n) time, O(m*n) space'
        }
    }