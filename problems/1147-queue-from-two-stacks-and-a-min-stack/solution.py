def process_operations(operations):
    # Queue using two stacks
    in_stack = []
    out_stack = []

    # Min-stack
    stack = []
    min_stack = []

    result = []

    for op in operations:
        if op[0] == "enqueue":
            in_stack.append(op[1])

        elif op[0] == "dequeue":
            # Move elements only when needed
            if not out_stack:
                while in_stack:
                    out_stack.append(in_stack.pop())

            result.append(out_stack.pop())

        elif op[0] == "mpush":
            value = op[1]
            stack.append(value)

            if not min_stack:
                min_stack.append(value)
            else:
                min_stack.append(min(value, min_stack[-1]))

        elif op[0] == "mpop":
            result.append(stack.pop())
            min_stack.pop()

        elif op[0] == "mtop":
            result.append(stack[-1])

        elif op[0] == "mmin":
            result.append(min_stack[-1])

    return result