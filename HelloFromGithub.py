def HelloFromGitHub(name:str):
    text = f"Hello {name}\n, hope you are doing good."
    text += "I'm GitHub, who welcomes you in my Codespace feature. \nHope you have a great experience."
    return text

if __name__ == "__main__":
    name = input("What is your name? ")
    print(HelloFromGitHub(name))