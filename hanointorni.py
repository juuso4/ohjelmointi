def get_valid_input():
    while True:
        try:
            number_of_disks = int(input("Enter the number of disks: "))
            if number_of_disks <= 0:
                print("Enter a positive number")
                continue

            if number_of_disks > 20:
                moves = 2 ** number_of_disks - 1
                response = input(
                    f"Warning: {number_of_disks} disks will require {moves} moves. Continue? (y/n): ").lower()
                if response != 'y':
                    continue

            return number_of_disks

        except ValueError:
            print("Enter a valid integer")


def print_towers(A, B, C):
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"C: {C}")
    print()


def move(n, source, auxiliary, target, A, B, C):
    if n <= 0:
        return

    move(n - 1, source, target, auxiliary, A, B, C)
    target.append(source.pop())
    print_towers(A, B, C)
    move(n - 1, auxiliary, source, target, A, B, C)


def tower_of_hanoi():

    number_of_disks = get_valid_input()


    A = list(range(number_of_disks, 0, -1))  # Source tower
    B = []  # Auxiliary tower
    C = []  # Target tower

    print("\nInitial state:")
    print_towers(A, B, C)

    print("Solving Tower of Hanoi...")
    move(number_of_disks, A, B, C, A, B, C)

    print("Puzzle solved!")
    print(f"Total moves: {2 ** number_of_disks - 1}")

if __name__ == "__main__":
    print("Welcome to Tower of Hanoi solver!")
    tower_of_hanoi()


