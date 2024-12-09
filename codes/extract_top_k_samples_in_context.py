import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(jsonl_file, top_k=2):
    prompts = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            record = json.loads(line)
            prompts.append(record['prompt'])

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(prompts)

    cosine_sim = cosine_similarity(tfidf_matrix)

    top_matches = {}
    for i, row in enumerate(cosine_sim):
        similarity_scores = list(enumerate(row))

        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        # Take top k (excluding self-match)
        top_k_scores = similarity_scores[1:1+top_k]

        top_matches[prompts[i]] = [(prompts[j], score) for j, score in top_k_scores]

    return top_matches


# Example Usage
jsonl_file_path = "../Dataset/processed_safim/api_completion_processed.jsonl"
matches = calculate_similarity(jsonl_file_path)

# Print the results
for prompt, top_2 in matches.items():
    print(f"Prompt: {prompt}")
    for match, score in top_2:
        print(f"  Match: {match}, Similarity: {score:.4f}")
    print()
