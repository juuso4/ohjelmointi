import time

def get_valid_input():
    while True:
        try:
            number_of_disks = int(input("Enter the number of disks: "))
            if number_of_disks <= 0:
                print("Enter a positive number")
                continue

            if number_of_disks > 19:
                moves = 2 ** number_of_disks - 1
                response = input(
                    f"Warning: {number_of_disks} disks will require {moves} moves. Continue? (y/n): ").lower()
                if response != 'y':
                    continue

            return number_of_disks

        except ValueError:
            print("Enter a valid integer")


def print_towers(A, B, C, move_count, start_time):
    current_time = time.time() - start_time
    print(f"\nMove #{move_count} (Time elapsed: {current_time:.2f} seconds)")
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"C: {C}")


def move(n, source, auxiliary, target, A, B, C, move_counter, start_time):
    if n <= 0:
        return move_counter

    move_counter = move(n - 1, source, target, auxiliary, A, B, C, move_counter, start_time)
    target.append(source.pop())
    move_counter += 1
    print_towers(A, B, C, move_counter, start_time)
    move_counter = move(n - 1, auxiliary, source, target, A, B, C, move_counter, start_time)

    return move_counter


def tower_of_hanoi():
    number_of_disks = get_valid_input()

    A = list(range(number_of_disks, 0, -1))
    B = []
    C = []

    print("\nInitial state:")
    start_time = time.time()
    print_towers(A, B, C, 0, start_time)

    print("\nSolving Tower of Hanoi...")
    total_moves = move(number_of_disks, A, B, C, A, B, C, 0, start_time)

    end_time = time.time()
    total_time = end_time - start_time

    print("\nPuzzle solved!")
    print(f"Total moves: {total_moves}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per move: {(total_time / total_moves):.4f} seconds")

    theoretical_moves = 2 ** number_of_disks - 1
    print(f"Theoretical minimum moves: {theoretical_moves}")


if __name__ == "__main__":
    print("Welcome to Tower of Hanoi solver")
    print("Note: The number of moves required is 2^n-1, where n is the number of disks")
    tower_of_hanoi()
