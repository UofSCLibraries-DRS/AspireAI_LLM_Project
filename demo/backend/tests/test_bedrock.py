from dotenv import load_dotenv
from src.api.chatbots.bedrock import BedrockChatbot


########################################################################
###                                                                  ###
### Run from backend root with `uv run python -m tests.test_bedrock` ###
###                                                                  ###
########################################################################

# Load AWS credentials from .env file
# Must be done BEFORE creating the BedrockChatbot
load_dotenv()


def main():
    # Path to M9 config
    config_path = "configs/chatbots/m9_bedrock.yaml"

    # Initialize the chatbot
    print("Initializing M9 Bedrock Chatbot...")
    m9_bot = BedrockChatbot(id="m9", config_path=config_path)

    print(f"Model ID: {m9_bot.config.model_id}")
    print(f"Temperature: {m9_bot.config.model_temperature}")
    print(f"Prompt Template: {m9_bot.prompt_template}")
    print(f"Stop Sequences: {m9_bot.stop_sequences}")
    print("-" * 50)

    # Test query
    test_prompt = "Hello, how are you today?"
    print(f"\nUser: {test_prompt}")

    # Generate response
    response = m9_bot.generate(test_prompt)
    print(f"Bot: {response}")
    print("-" * 50)

    # Optional: Test another query
    test_prompt_2 = "What organizations were John H. McCray affiliated with?"
    print(f"\nUser: {test_prompt_2}")
    response_2 = m9_bot.generate(test_prompt_2)
    print(f"Bot: {response_2}")


if __name__ == "__main__":
    main()
