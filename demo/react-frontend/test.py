from gaico import Experiment

# Sample LLM responses comparing different models
llm_responses = {
    "Google": "Title: Jimmy Kimmel Reacts to Donald Trump Winning...",
    "Mixtral 8x7b": "I'm an AI and I don't have the ability to predict...",
    "SafeChat": "Sorry, I am designed not to answer such a question.",
}

# Initialize and run comparison
exp = Experiment(llm_responses=llm_responses, reference_answer=None)
results = exp.compare(
    # metrics=['Jaccard', 'ROUGE'],
    plot=True,
    output_csv_path="experiment_report.csv"
)

print(results)