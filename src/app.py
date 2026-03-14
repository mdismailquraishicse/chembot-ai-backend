from chembot_ai import ChatBotAI

bot = ChatBotAI()

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    answer = bot.ask(question=question)
    print("\nChemBot:", answer)
    print()