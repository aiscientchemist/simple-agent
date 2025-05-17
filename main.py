import os
import json
from datetime import datetime
from dotenv import load_dotenv

# For GitHub
from github import Github

# For Reddit
import praw

# For data handling
import pandas as pd

# For AI (Hugging Face)
from transformers import pipeline


load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")


g = Github(GITHUB_TOKEN)

# Reddit (needs your client ID, secret, username, and password)
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD,
    # CHANGE 'my_awesome_user' to your actual Reddit username
    user_agent="InsightAgent by YourRedditUsername (e.g., InsightAgent by u/my_awesome_user)"
)

# AI Models (downloaded automatically the first time)
qa_pipeline = pipeline("question-answering",
                       model="distilbert-base-cased-distilled-squad")
# You can try other models like "deepset/roberta-base-squad2" if this one isn't precise enough

# --- Data Storage Folder ---
DATA_DIR = "data"
# Creates the 'data' folder if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# --- Helper Functions (We will add these in the next steps) ---


def fetch_github_data(query, limit=10):
    print(f"Fetching GitHub data for '{query}'...")
    repos_data = []
    try:
        # Search for repositories
        for repo in g.search_repositories(query, sort="stars", order="desc"):
            if len(repos_data) >= limit:
                break
            try:
                readme_content = ""
                # Try to get README content - some repos might not have it or it might be too big
                try:
                    readme = repo.get_readme().decoded_content.decode('utf-8')
                    readme_content = readme
                except Exception as e:
                    # print(f"Could not get README for {repo.full_name}: {e}")
                    pass  # It's okay if a README is missing

                repos_data.append({
                    "name": repo.full_name,
                    "description": repo.description,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "language": repo.language,
                    "url": repo.html_url,
                    "readme_content": readme_content,
                    "topics": repo.get_topics()  # Get topics/tags
                })
                print(f"  - Added {repo.full_name}")
            except Exception as e:
                print(f"Error processing repo {repo.full_name}: {e}")
                continue
    except Exception as e:
        print(f"Error fetching GitHub data: {e}")

    file_path = os.path.join(
        DATA_DIR, f"github_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(repos_data, f, ensure_ascii=False, indent=4)
    print(f"Saved GitHub data to {file_path}")
    return file_path


def fetch_reddit_data(subreddit_name, query, limit=10):
    print(f"Fetching Reddit data from r/{subreddit_name} for '{query}'...")
    posts_data = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        # Search for posts within the subreddit
        for submission in subreddit.search(query, limit=limit):
            comments_text = []
            # Get top comments (simplified)
            # Remove "more comments" links
            submission.comments.replace_more(limit=0)
            # Get top 3 comments
            for top_level_comment in submission.comments.list()[:3]:
                if hasattr(top_level_comment, 'body'):
                    comments_text.append(top_level_comment.body)

            posts_data.append({
                "title": submission.title,
                "score": submission.score,
                "url": submission.url,
                "selftext": submission.selftext,  # The main post text
                # Join comments into one string
                "comments": "\n".join(comments_text)
            })
            print(f"  - Added Reddit post: {submission.title}")
    except Exception as e:
        print(f"Error fetching Reddit data: {e}")

    file_path = os.path.join(
        DATA_DIR, f"reddit_{subreddit_name}_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(posts_data, f, ensure_ascii=False, indent=4)
    print(f"Saved Reddit data to {file_path}")
    return file_path


def load_data_from_file(file_path):
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def analyze_data(df, source_type, library_name=None):
    if df.empty:
        print("No data to analyze.")
        return

    print(f"\n--- Analysis Report ({source_type}) ---")

    if source_type == 'github':

        print("\nTop 5 Repositories by Stars:")
        for index, row in df.nlargest(5, 'stars').iterrows():
            print(
                f"- {row['name']} ({row['stars']} stars): {row['description'][:100]}...")

        if library_name:
            print(f"\nProjects mentioning '{library_name}':")
            df['text_content'] = df['description'].fillna(
                '') + ' ' + df['readme_content'].fillna('')
            df['mentions'] = df['text_content'].str.lower(
            ).str.count(library_name.lower())

            mentions_df = df[df['mentions'] > 0].sort_values(
                by='mentions', ascending=False)
            if not mentions_df.empty:
                for index, row in mentions_df.head(5).iterrows():
                    print(
                        f"- {row['name']} (Mentions: {row['mentions']}) - {row['description'][:100]}...")
            else:
                print(f"  No projects found mentioning '{library_name}'.")

    elif source_type == 'reddit':

        print("\nTop 5 Reddit Posts by Score:")
        for index, row in df.nlargest(5, 'score').iterrows():
            print(f"- {row['title']} ({row['score']} score) - {row['url']}")

        if library_name:
            print(f"\nReddit posts mentioning '{library_name}':")
            df['text_content'] = df['title'].fillna(
                '') + ' ' + df['selftext'].fillna('') + ' ' + df['comments'].fillna('')
            df['mentions'] = df['text_content'].str.lower(
            ).str.count(library_name.lower())

            mentions_df = df[df['mentions'] > 0].sort_values(
                by='mentions', ascending=False)
            if not mentions_df.empty:
                for index, row in mentions_df.head(5).iterrows():
                    print(
                        f"- {row['title']} (Mentions: {row['mentions']}) - {row['url']}")
            else:
                print(f"  No posts found mentioning '{library_name}'.")


def ask_ai_question(question, context_text):
    if not context_text:
        print("No context provided for the AI to answer. Please provide a relevant text.")
        return "Sorry, I don't have enough information to answer that."

    print(f"\nAsking AI: '{question}'...")
    try:

        max_context_length = 500
        trimmed_context = context_text[:max_context_length] + "..." if len(
            context_text) > max_context_length else context_text

        result = qa_pipeline(question=question, context=trimmed_context)
        answer = result['answer']
        score = result['score']
        print(f"AI Answer: {answer} (Confidence: {score:.2f})")
        return answer
    except Exception as e:
        print(f"Error asking AI question: {e}")
        return "Sorry, I encountered an error trying to answer that question."


def main():
    print("Welcome to InsightAgent!")
    print("Use 'fetch' to get data, 'analyze' to get insights, 'ask' to query the AI.")
    print("Example: python agent.py fetch github 'LLM applications'")
    print("Example: python agent.py analyze data/github_LLM_applications_*.json 'transformers'")
    print("Example: python agent.py ask 'What is this project about?' 'data/github_LLM_applications_*.json'")

    import sys
    args = sys.argv[1:]
    if not args:
        print("Please provide a command: fetch, analyze, or ask.")
        return

    command = args[0]

    if command == "fetch":
        if len(args) < 3:
            print(
                "Usage: python agent.py fetch <source> <query> [subreddit] [limit]")
            print("  <source>: 'github' or 'reddit'")
            print("  <query>: search term")
            print(
                "  [subreddit]: (optional, for reddit) subreddit name (e.g., MachineLearning)")
            print("  [limit]: (optional) number of items to fetch (default 10)")
            return

        source = args[1].lower()
        query = args[2]
        subreddit_name = None
        limit = 10
        if source == 'reddit' and len(args) > 3:
            subreddit_name = args[3]
        if (source == 'github' and len(args) > 3) or (source == 'reddit' and len(args) > 4):
            try:
                if source == 'github':
                    limit = int(args[3])
                elif source == 'reddit':
                    limit = int(args[4])
            except ValueError:
                print("Limit must be a number.")
                return

        if source == "github":
            fetch_github_data(query, limit)
        elif source == "reddit" and subreddit_name:
            fetch_reddit_data(subreddit_name, query, limit)
        else:
            print(
                "Invalid source or missing subreddit for Reddit. Use 'github' or 'reddit <subreddit_name>'.")

    elif command == "analyze":
        if len(args) < 2:
            print(
                "Usage: python agent.py analyze <data_file_path> [library_name]")
            print("  <data_file_path>: Path to a JSON file generated by 'fetch' (e.g., data/github_LLM_applications_2023....json)")
            print(
                "  [library_name]: (optional) Specific library to look for (e.g., 'transformers', 'pytorch')")
            return

        data_file_path = args[1]
        library_name = None
        if len(args) > 2:
            library_name = args[2]

        source_type = 'github' if 'github' in data_file_path.lower() else 'reddit'

        df = load_data_from_file(data_file_path)
        analyze_data(df, source_type, library_name)

    elif command == "ask":
        if len(args) < 3:
            print(
                "Usage: python agent.py ask <question_in_quotes> <data_file_path_or_text_index>")
            print(
                "  <question_in_quotes>: The question you want to ask (e.g., '\"What is this project about?\"')")
            print("  <data_file_path_or_text_index>: Path to a JSON data file from which to extract context, OR a specific index if the file has multiple entries.")
            print(
                "    If a file path, it will try to use the description/selftext/readme from the first entry.")
            print("    If a file path with a number (e.g., 'data/github_...json 0'), it will use that specific entry.")
            return

        question = args[1]
        # This can be a file path or file path + index
        context_source = args[2]

        context_text = ""
        if os.path.exists(context_source):  # It's a file path
            df = load_data_from_file(context_source)
            if not df.empty:

                entry_index = 0
                if len(args) > 3:
                    try:
                        entry_index = int(args[3])
                    except ValueError:
                        print("Invalid index provided. Using default index 0.")

                if entry_index < len(df):
                    if 'readme_content' in df.columns:  # GitHub
                        context_text = df.loc[entry_index,
                                              'readme_content'] or df.loc[entry_index, 'description']
                    elif 'selftext' in df.columns:  # Reddit
                        context_text = df.loc[entry_index,
                                              'selftext'] or df.loc[entry_index, 'title']
                    else:
                        print(
                            "Could not find relevant text content in the loaded data.")
                else:
                    print(
                        f"Index {entry_index} is out of bounds for the loaded data (max {len(df)-1}).")
            else:
                print("No data loaded from the file to provide context.")
        else:  # It's not a file path, assume it's a text index
            print("Please provide a path to a data file for context.")

        if context_text:
            ask_ai_question(question, context_text)
        else:
            print("No suitable context could be determined for the AI question.")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: fetch, analyze, ask")


if __name__ == "__main__":
    main()
