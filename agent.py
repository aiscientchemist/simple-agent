import os
import json
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import sys  # For sys.argv

# For GitHub
from github import Github, GithubException

# For Reddit
import praw
from prawcore.exceptions import PrawcoreException

# For AI (Hugging Face)
from transformers import pipeline

# For S3
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# --- Load Environment Variables ---
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# --- Initialize Services ---
g = None
if GITHUB_TOKEN:
    try:
        g = Github(GITHUB_TOKEN)
        g.get_user().login  # Test connection
        print("Successfully connected to GitHub.")
    except GithubException as e:
        print(
            f"Warning: GitHub connection failed: {e}. GitHub functionality will be limited.")
        g = None
    except Exception as e:  # Catch any other potential errors during init
        print(
            f"Warning: An unexpected error occurred initializing GitHub client: {e}. GitHub functionality will be limited.")
        g = None
else:
    print("Warning: GITHUB_TOKEN not found. GitHub functionality disabled.")

reddit = None
if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USERNAME and REDDIT_PASSWORD:
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            username=REDDIT_USERNAME,
            password=REDDIT_PASSWORD,
            # Personalized user agent
            user_agent=f"InsightAgent_v1 by u/{REDDIT_USERNAME}"
        )
        reddit.user.me()  # Test connection
        print("Successfully connected to Reddit.")
    except PrawcoreException as e:
        print(
            f"Warning: Reddit connection failed: {e}. Reddit functionality will be limited.")
        reddit = None
    except Exception as e:  # Catch any other potential errors during init
        print(
            f"Warning: An unexpected error occurred initializing Reddit client: {e}. Reddit functionality will be limited.")
        reddit = None
else:
    print("Warning: Reddit credentials not found. Reddit functionality disabled.")

# AI Models (Downloads on first use if not cached)
try:
    qa_pipeline = pipeline("question-answering",
                           model="distilbert-base-cased-distilled-squad")
    print("QA pipeline initialized.")
except Exception as e:
    print(
        f"Error initializing QA pipeline: {e}. AI Q&A functionality will be disabled.")
    qa_pipeline = None


# S3 Client Initialization
s3_client = None
if S3_BUCKET_NAME:
    try:
        s3_client = boto3.client('s3')
        # Test if bucket exists and we have access
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        print(f"S3 client initialized. Target bucket: {S3_BUCKET_NAME}")
    except (NoCredentialsError, PartialCredentialsError):
        print("AWS S3 credentials not found. S3 functionality will be DISABLED.")
        s3_client = None
    except ClientError as e:
        if e.response['Error']['Code'] == '404':  # Not Found
            print(
                f"S3 Bucket '{S3_BUCKET_NAME}' not found. S3 functionality will be DISABLED.")
        elif e.response['Error']['Code'] == '403':  # Forbidden
            print(
                f"Access denied to S3 Bucket '{S3_BUCKET_NAME}'. Check IAM permissions. S3 functionality will be DISABLED.")
        else:
            print(f"S3 ClientError: {e}. S3 functionality will be DISABLED.")
        s3_client = None
    except Exception as e:
        print(
            f"An unexpected error occurred initializing S3 client: {e}. S3 functionality will be DISABLED.")
        s3_client = None
else:
    print("S3_BUCKET_NAME not set in .env file. S3 functionality will be DISABLED.")

# Data Storage Configuration
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def sanitize_for_filename(text):
    """Sanitizes text to be safe for filenames/S3 keys."""
    return "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in text).strip('_').lower()[:50]


def save_data_to_storage(data_to_save, source_type, query_details):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    safe_query_details = sanitize_for_filename(query_details)

    base_filename = f"{source_type}_{safe_query_details}_{timestamp}.json"
    s3_key_prefix = f"{source_type}_data/"
    s3_key = f"{s3_key_prefix}{base_filename}"

    if s3_client and S3_BUCKET_NAME:
        try:
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                Body=json.dumps(data_to_save, ensure_ascii=False, indent=4),
                ContentType='application/json'
            )
            s3_uri = f"s3://{S3_BUCKET_NAME}/{s3_key}"
            print(f"Successfully saved data to S3: {s3_uri}")
            return s3_uri
        except ClientError as e:
            print(f"S3 Save Error: {e}. Falling back to local storage.")
        except Exception as e:  # Catch any other unexpected errors
            print(
                f"An unexpected error occurred during S3 save: {e}. Falling back to local storage.")

    local_file_path = os.path.join(DATA_DIR, base_filename)
    try:
        with open(local_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved data locally: {local_file_path}")
        return local_file_path
    except Exception as e:
        print(
            f"Local Save Error: Failed to save data locally to {local_file_path}: {e}")
        return None


def load_data_from_storage(storage_path):
    print(f"Attempting to load data from: {storage_path}")
    if storage_path.startswith("s3://"):
        if not s3_client or not S3_BUCKET_NAME:
            print(
                "S3 client not configured or S3_BUCKET_NAME missing. Cannot load from S3 path.")
            return pd.DataFrame()
        try:
            path_parts = storage_path.replace("s3://", "").split("/")
            bucket = path_parts[0]
            key = "/".join(path_parts[1:])
            if not bucket or not key:  # Basic validation
                print(f"Invalid S3 path format: {storage_path}")
                return pd.DataFrame()

            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)
            print(f"Successfully loaded data from S3: {storage_path}")
            return pd.DataFrame(data)
        except ClientError as e:
            print(
                f"S3 Load Error: Could not load data from {storage_path}. Error: {e}")
            return pd.DataFrame()
        except Exception as e:  # Catch any other unexpected errors
            print(
                f"An unexpected error occurred during S3 load from {storage_path}: {e}")
            return pd.DataFrame()
    else:
        if not os.path.exists(storage_path):
            print(f"Local Load Error: File not found at {storage_path}")
            return pd.DataFrame()
        try:
            with open(storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded data locally from: {storage_path}")
            return pd.DataFrame(data)
        except Exception as e:
            print(
                f"Local Load Error: Failed to load data from {storage_path}: {e}")
            return pd.DataFrame()


def fetch_github_data(query, limit=10):
    if not g:
        print("GitHub client not initialized. Skipping GitHub fetch.")
        return None
    print(f"Fetching GitHub data for '{query}' (limit: {limit})...")
    repos_data = []
    try:
        repositories = g.search_repositories(
            query=query, sort="stars", order="desc")
        for i, repo in enumerate(repositories):
            if i >= limit:
                break
            readme_content = ""
            try:
                readme_file = repo.get_readme()
                readme_content = readme_file.decoded_content.decode(
                    'utf-8', errors='replace')
            except GithubException:  # Handles 404 if README doesn't exist or other API errors
                # print(f"  - No README found for {repo.full_name} or error fetching it.")
                pass
            except Exception as e_readme:
                print(
                    f"  - Error decoding README for {repo.full_name}: {e_readme}")

            repos_data.append({
                "id": repo.id,
                "name": repo.full_name,
                "description": repo.description,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "language": repo.language,
                "url": repo.html_url,
                "created_at": repo.created_at.isoformat() if repo.created_at else None,
                "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                "topics": repo.get_topics(),
                # Limit README size for practical reasons
                "readme_content": readme_content[:200000]
            })
            print(
                f"  - Added GitHub repo: {repo.full_name} ({repo.stargazers_count} stars)")
    except GithubException as e:
        print(f"GitHub API Error during fetch: {e}")
        if not repos_data:
            return None  # If error occurred before fetching any data
    except Exception as e:
        print(f"An unexpected error occurred during GitHub fetch: {e}")
        if not repos_data:
            return None

    if not repos_data:
        print("No GitHub data fetched.")
        return None

    storage_path = save_data_to_storage(repos_data, "github", query)
    return storage_path


def fetch_reddit_data(subreddit_name, query, limit=10, sort_type="relevance", time_filter="all"):
    if not reddit:
        print("Reddit client not initialized. Skipping Reddit fetch.")
        return None
    print(
        f"Fetching Reddit data from r/{subreddit_name} for '{query}' (limit: {limit}, sort: {sort_type}, time: {time_filter})...")
    posts_data = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.search(query, limit=limit, sort=sort_type, time_filter=time_filter):
            comments_sample = []
            try:
                # Resolve MoreComments objects
                submission.comments.replace_more(limit=0)
                # Get top 3 comments
                for top_level_comment in submission.comments.list()[:3]:
                    if hasattr(top_level_comment, 'body') and top_level_comment.body != '[deleted]' and top_level_comment.body != '[removed]':
                        # Limit comment length
                        comments_sample.append(top_level_comment.body[:500])
            except Exception as e_comm:
                print(
                    f"  - Error fetching comments for post '{submission.title}': {e_comm}")

            posts_data.append({
                "id": submission.id,
                "title": submission.title,
                "score": submission.score,
                "url": submission.url,
                "permalink": f"https://www.reddit.com{submission.permalink}",
                "selftext": submission.selftext,
                "created_utc": submission.created_utc,
                "num_comments_total": submission.num_comments,
                "author": submission.author.name if submission.author else "[deleted]",
                "flair": submission.link_flair_text,
                "comments_sample": "\n---\n".join(comments_sample)
            })
            print(
                f"  - Added Reddit post: {submission.title} (Score: {submission.score})")
    except PrawcoreException as e:
        print(f"Reddit API Error during fetch: {e}")
        if not posts_data:
            return None
    except Exception as e:
        print(f"An unexpected error occurred during Reddit fetch: {e}")
        if not posts_data:
            return None

    if not posts_data:
        print("No Reddit data fetched.")
        return None

    storage_path = save_data_to_storage(
        posts_data, "reddit", f"{subreddit_name}_{query}")
    return storage_path


def analyze_data(df, source_type, library_name=None):
    if df.empty:
        print("No data to analyze (DataFrame is empty).")
        return

    print(f"\n--- Analysis Report ({source_type}) ---")

    if source_type == 'github':
        if 'stars' in df.columns:
            print("\nTop 5 Repositories by Stars:")
            for _, row in df.nlargest(5, 'stars').iterrows():
                print(
                    f"- {row.get('name', 'N/A')} ({row.get('stars', 0)} stars): {str(row.get('description', ''))[:100]}...")
        else:
            print("Column 'stars' not found for GitHub analysis.")

        if library_name:
            print(f"\nProjects mentioning '{library_name}':")
            # Combine description and readme for search
            df['text_for_search'] = df['description'].fillna(
                '') + ' ' + df['readme_content'].fillna('')
            df['mentions'] = df['text_for_search'].str.lower(
            ).str.count(library_name.lower())

            mentions_df = df[df['mentions'] > 0].sort_values(
                by='mentions', ascending=False)
            if not mentions_df.empty:
                for _, row in mentions_df.head(5).iterrows():
                    print(
                        f"- {row.get('name', 'N/A')} (Mentions: {row.get('mentions', 0)})")
            else:
                print(f"  No projects found mentioning '{library_name}'.")

    elif source_type == 'reddit':
        if 'score' in df.columns:
            print("\nTop 5 Reddit Posts by Score:")
            for _, row in df.nlargest(5, 'score').iterrows():
                print(
                    f"- {row.get('title', 'N/A')} ({row.get('score', 0)} score) - {row.get('url', '')}")
        else:
            print("Column 'score' not found for Reddit analysis.")

        if library_name:
            print(f"\nReddit posts mentioning '{library_name}':")
            df['text_for_search'] = df['title'].fillna(
                '') + ' ' + df['selftext'].fillna('') + ' ' + df['comments_sample'].fillna('')
            df['mentions'] = df['text_for_search'].str.lower(
            ).str.count(library_name.lower())

            mentions_df = df[df['mentions'] > 0].sort_values(
                by='mentions', ascending=False)
            if not mentions_df.empty:
                for _, row in mentions_df.head(5).iterrows():
                    print(
                        f"- {row.get('title', 'N/A')} (Mentions: {row.get('mentions', 0)})")
            else:
                print(f"  No posts found mentioning '{library_name}'.")
    else:
        print(f"Unknown source type for analysis: {source_type}")


def ask_ai_question(question, context_text):
    if not qa_pipeline:
        print("QA pipeline not initialized. Cannot answer question.")
        return "AI Q&A is currently unavailable."
    if not context_text or not isinstance(context_text, str) or len(context_text.strip()) == 0:
        print("No valid context provided for the AI to answer.")
        return "Sorry, I don't have enough information (empty context) to answer that."

    # Max context length for distilbert is 512 tokens.
    # A simple approximation: characters. 1 token ~ 4 chars. So ~2000 chars.
    # For more accuracy, use tokenizer.encode then truncate.
    # For simplicity, we'll just pass it and let the pipeline handle truncation if needed,
    # or use a simple character limit for the input to the pipeline.
    # The pipeline itself should handle overly long contexts by truncating.
    # However, to be safe and manage memory/performance:
    max_chars_for_context = 4000  # Rough estimate
    trimmed_context = context_text[:max_chars_for_context]

    print(f"\nAsking AI: '{question}'...")
    try:
        result = qa_pipeline(question=question, context=trimmed_context)
        answer = result['answer']
        score = result.get('score', 0.0)  # Get score if available
        print(f"AI Answer: {answer} (Confidence: {score:.2f})")
        return answer
    except Exception as e:
        print(f"Error during AI question answering: {e}")
        return "Sorry, I encountered an error trying to answer that question."


def main():
    print("Welcome to InsightAgent!")
    print("-------------------------")

    args = sys.argv[1:]

    if not args:
        print("Usage: python agent.py <command> [options...]")
        print("Commands: fetch, analyze, ask")
        print("  fetch github <query> [limit]")
        print(
            "  fetch reddit <subreddit> <query> [limit] [sort_type] [time_filter]")
        print("  analyze <storage_path_or_local_file> [library_name]")
        print(
            "  ask \"<question>\" <storage_path_or_local_file> [index_of_entry]")
        return

    command = args[0].lower()

    if command == "fetch":
        if len(args) < 3:
            print("Usage: python agent.py fetch <source> <details...>")
            return
        source = args[1].lower()
        if source == "github":
            if len(args) < 3:
                print("Usage: python agent.py fetch github <query> [limit]")
                return
            query = args[2]
            limit = int(args[3]) if len(args) > 3 and args[3].isdigit() else 10
            storage_path = fetch_github_data(query, limit)
        elif source == "reddit":
            if len(args) < 4:
                print(
                    "Usage: python agent.py fetch reddit <subreddit> <query> [limit] [sort_type] [time_filter]")
                print(
                    "  sort_type (optional): relevance, hot, top, new, comments (default: relevance)")
                print(
                    "  time_filter (optional for top/comments sort): hour, day, week, month, year, all (default: all)")
                return
            subreddit_name = args[2]
            query = args[3]
            limit = int(args[4]) if len(args) > 4 and args[4].isdigit() else 10
            sort_type = args[5] if len(args) > 5 else "relevance"
            time_filter = args[6] if len(args) > 6 else "all"
            storage_path = fetch_reddit_data(
                subreddit_name, query, limit, sort_type, time_filter)
        else:
            print(
                f"Invalid source for fetch: {source}. Use 'github' or 'reddit'.")
            return

        if storage_path:
            print(f"\nFetch complete. Data stored at: {storage_path}")
        else:
            print("\nFetch operation failed or produced no data.")

    elif command == "analyze":
        if len(args) < 2:
            print(
                "Usage: python agent.py analyze <storage_path_or_local_file> [library_name]")
            return
        storage_path = args[1]
        library_name = args[2] if len(args) > 2 else None

        df = load_data_from_storage(storage_path)
        if not df.empty:
            source_type = 'github' if 'github' in storage_path.lower(
            ) else 'reddit' if 'reddit' in storage_path.lower() else 'unknown'
            analyze_data(df, source_type, library_name)
        else:
            print(
                f"Failed to load data from '{storage_path}' or data is empty.")

    elif command == "ask":
        if len(args) < 3:
            print(
                "Usage: python agent.py ask \"<question>\" <storage_path_or_local_file> [index_of_entry]")
            return
        question = args[1]
        storage_path = args[2]
        entry_index = int(args[3]) if len(
            args) > 3 and args[3].isdigit() else 0

        df = load_data_from_storage(storage_path)
        if not df.empty:
            if 0 <= entry_index < len(df):
                entry = df.iloc[entry_index]
                context_text = ""
                # Prioritize more descriptive fields for context
                if 'readme_content' in entry and pd.notna(entry['readme_content']) and entry['readme_content'].strip():
                    context_text = entry['readme_content']
                elif 'selftext' in entry and pd.notna(entry['selftext']) and entry['selftext'].strip():
                    context_text = entry['selftext']
                elif 'description' in entry and pd.notna(entry['description']) and entry['description'].strip():
                    context_text = entry['description']
                # Fallback to title if others are empty
                elif 'title' in entry and pd.notna(entry['title']) and entry['title'].strip():
                    context_text = entry['title']

                if context_text:
                    ask_ai_question(question, context_text)
                else:
                    print(
                        f"No suitable text context found in entry {entry_index} from {storage_path}.")
            else:
                print(
                    f"Index {entry_index} is out of bounds for data from {storage_path} (Loaded {len(df)} entries).")
        else:
            print(
                f"Failed to load data from '{storage_path}' or data is empty.")
    else:
        print(
            f"Unknown command: '{command}'. Use 'fetch', 'analyze', or 'ask'.")


if __name__ == "__main__":
    main()
