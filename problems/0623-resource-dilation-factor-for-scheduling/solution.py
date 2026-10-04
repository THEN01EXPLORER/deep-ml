def compute_dilation_factor(tasks: list, capacity: int) -> dict:
    """
    Compute the resource dilation factor for a constrained scheduling problem.
    """
    n = len(tasks)

    from functools import lru_cache

    # Calculate critical path
    @lru_cache(None)
    def critical_path(i):
        successors = [
            j for j in range(n)
            if i in tasks[j]["dependencies"]
        ]

        if not successors:
            return tasks[i]["duration"]

        return tasks[i]["duration"] + max(
            critical_path(j) for j in successors
        )

    cp = max(
        (critical_path(i) for i in range(n)),
        default=0.0
    )

    # Calculate total work
    total_work = sum(
        task["duration"] * task["resources"]
        for task in tasks
    )

    # Force float output
    lower_bound = float(max(cp, total_work / capacity))

    # Greedy scheduling
    start_times = [0.0] * n
    finish_times = [0.0] * n

    scheduled = set()
    completed = set()
    running = {}

    current_time = 0.0
    used_resources = 0

    while len(completed) < n:

        # Complete finished tasks
        for i in list(running):
            if running[i] <= current_time:
                used_resources -= tasks[i]["resources"]
                completed.add(i)
                del running[i]

        # Schedule ready tasks in index order
        for i in range(n):
            if i in scheduled:
                continue

            task = tasks[i]

            if not all(dep in completed for dep in task["dependencies"]):
                continue

            if used_resources + task["resources"] <= capacity:
                start_times[i] = float(current_time)
                finish_times[i] = float(
                    current_time + task["duration"]
                )

                running[i] = finish_times[i]
                used_resources += task["resources"]
                scheduled.add(i)

        if len(completed) == n:
            break

        # Jump to next task completion
        if running:
            current_time = min(running.values())
        else:
            raise ValueError("Invalid dependencies or resource requirements")

    actual_makespan = float(max(finish_times, default=0.0))
    dilation_factor = float(
        actual_makespan / lower_bound
        if lower_bound > 0
        else 0.0
    )

    return {
        "dilation_factor": round(dilation_factor, 4),
        "actual_makespan": round(actual_makespan, 4),
        "lower_bound": round(lower_bound, 4),
        "start_times": [round(t, 4) for t in start_times]
    }