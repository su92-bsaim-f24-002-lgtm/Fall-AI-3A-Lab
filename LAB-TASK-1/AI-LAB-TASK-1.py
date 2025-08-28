ABDULLAH
SU92-BSAIM-F24-02
BSAI-3A

class HangmanGame:
    def __init__(self, word):
        self.word = word.upper()                          
        self.lives = 6    
        self.guessed_letters = []                      
        self.hangman_stages = [                
            """
               ------
               |    |
               |    O
               |   /|\\
               |   / \\
               |
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |   / 
               |
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |    
               |
            """,
            """
               ------
               |    |
               |    O
               |   /|
               |    
               |
            """,
            """
               ------
               |    |
               |    O
               |    |
               |    
               |
            """,
            """
               ------
               |    |
               |    O
               |    
               |    
               |
            """,
            """
               ------
               |    |
               |    
               |    
               |    
               |
            """
        ]

        def display_word(self):
            return " ".join([letter 
                if letter in self.guessed_letters 
                    else "_" for letter in self.word])
    
        def print_hangman(self):
            print(self.hangman_stages[self.lives])


    def guess_letter(self, letter):
        letter = letter.upper()
        if letter in self.guessed_letters:
            print(" You already guessed this letter!")
        elif letter in self.word:
            print(f" Good job! {letter} is in the word.")
            self.guessed_letters.append(letter)
        else:
            print(f" Wrong guess! {letter} is not in the word.")
            self.guessed_letters.append(letter)
            self.lives -= 1

        def won(self):
            return all(letter in self.guessed_letters 
                       for letter in self.word)

        def lost(self):
            return self.lives <= 0
        
        def run(self):
            print("Welcome to HangmanGame")
            while not self.is_won() and not self.is_lost():
                print("\nWord:", self.display_word())
            self.print_hangman()
            guess = input("Guess a letter: ").strip()
            
            if len(guess) != 1 or not guess.isalpha():
                print(" Please enter a single letter!")
               
            self.guess_letter(guess)

        if self.is_won():
            print("\n Congratulations! You guessed the word:", self.word)
        else:
            print("\n You lost! The word was:", self.word)
        self.print_hangman()

       
        game = HangmanGame("PYTHON")   
        game.run()
