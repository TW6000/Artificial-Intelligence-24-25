import argparse
import subprocess

#written to demonstrate difference between forward and backward chaining for rules-based AI BSU AI S1 TOM WHITE 

def main():
    parser = argparse.ArgumentParser(description='Execute either forward or backward chaining script.')
    parser.add_argument('--forward', action='store_true', help='Execute forward_chaining.py')
    parser.add_argument('--backward', action='store_true', help='Execute backward_chaining.py')
    
    args = parser.parse_args()
    
    if args.forward:
        subprocess.run(['python', 'forward_chaining.py'])
    elif args.backward:
        subprocess.run(['python', 'backward_chaining.py'])
    else:
        print("Please specify --forward or --backward argument to call the corresponding algorithm")

if __name__ == '__main__':
    main()
