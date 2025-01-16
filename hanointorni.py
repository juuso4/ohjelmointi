number_of_disks = 5
A = list(range(number_of_disks, 0, -1))
B = []
C = []

def move(n, source, auxliary, target):
    if n <= 0:
        return

    move(n - 1, source, target, auxliary)

    target.append(source.pop())

    print(A, B, C, '\n')

    move(n - 1, auxliary, source, target)

move(number_of_disks, A, B, C)