fileName = "species_name_temp.txt"

def main():
    with open(fileName, "r") as species_name:
        for line in species_name:
            print(line)
main()
