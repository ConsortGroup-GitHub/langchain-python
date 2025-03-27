from chain_factory import create_chat_chain

def main():
    chain = create_chat_chain(session_id="chat3")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = chain.invoke(user_input)
        print("Assistant:", response)

if __name__ == '__main__':
    main()
