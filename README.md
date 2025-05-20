InsightAgent => AI-Powered Data Analysis 

IndatsightAgent is a Python-based AI system designed to fetch a from GitHub and Reddit, process it, and leverage a local Hugging Face Language Model  to answer user queries about trends and insights within that data.



Features

	Data Retrieval:
    	  Fetch repository data from GitHub based on search queries (e.g., name, description, stars, forks, README content, topics).
    	  Fetch post data from Reddit subreddits based on search queries (e.g., title, selftext, score, top comments).
	Data Processing:
    	  Store fetched data locally in JSON format within a `data/` directory.
    	  Load and process data using Pandas DataFrames.
	Data Analysis:
    	 Basic analysis: Identify top repositories by stars, top Reddit posts by score.
         Keyword/Library Mention Counting: Count occurrences of specific terms (e.g., "transformers", "pytorch") within descriptions, READMEs, or post content.
	AI-Powered Q&A:
        Answer user questions about the fetched data using a local Question Answering model (`distilbert-base-cased-distilled-squad`) from Hugging Face Transformers.
	Interface:
       Command-Line Interface (CLI) for all operations.

 🛠️ Prerequisites

   Python 3.8+
  pip (Python package installer)
   Git (for cloning the repository)

 ⚙️ Setup & Installation

This guide assumes you have a copy of the project files. --> you have to do it :)

1.  Ensure Python is Installed:   if not You can download it from [python.org](https://www.python.org/downloads/). 
       You need Python 3.8 or newer. 
       During installation, make sure to check the box that says "Add Python to PATH" (or similar).

2.  Navigate to the Project Directory:
       Open your terminal or command prompt.
       Change directory (`cd`) to the folder where the `InsightAgent-Project` files are located (e.g., where `agent.py` is).
       Example: `cd path/to/your/InsightAgent-Project`

3.  Create a Virtual Environment (Highly Recommended):
    A virtual environment keeps the project's dependencies isolated.
    ```bash
    python -m venv venv
    ```
       Activate the environment:
           On Windows: `venv\Scripts\activate`
           On macOS/Linux: `source venv/bin/activate`
       (You should see `(venv)` at the beginning of your terminal prompt if it's active).*

4.  Install Dependencies:
    If a `requirements.txt` file is included in the project, run:
    ```bash
    pip install -r requirements.txt
    ```
    If there is no `requirements.txt` file, you will need to install the packages manually:
    ```bash
    pip install transformers torch pandas PyGithub praw python-dotenv
    ```
    (It's best practice to include a `requirements.txt` file. If you are the project owner, you can generate it in your activated virtual environment by running `pip freeze > requirements.txt` after installing all packages.)

 🔑 Configuration (API Keys)

This project requires API keys to access GitHub and Reddit. These keys are personal and should NOT be shared directly in the code or committed to Git.  like my are mine i have mine you have yours Each user needs to set up their own.

1.  Obtain API Keys:
       GitHub Token:
        1.  Go to your GitHub account settings.
        2.  Navigate to `Developer settings` -> `Personal access tokens` -> `Tokens (classic)`.
        3.  Click `Generate new token` (classic).
        4.  Give it a note (e.g., "InsightAgent Access").
        5.  Select scopes: at least `repo` (to read public repository information) and `read:org`.
        6.  Generate the token and copy it immediately. You won't see it again.
    Reddit Credentials:
        1.  Go to `reddit.com/prefs/apps`.
        2.  Scroll down and click `create another app` (or `create app`).
        3.  Fill in the details:
               name: e.g., `MyInsightAgentApp`
               type: Select `script`.
               redirect uri: `http://localhost:8080` (this is required even if not directly used by a browser).
        4.  Click `create app`.
        5.  Note down the `client ID` (it's under the app name, looks like a random string of characters) and the `secret`. You will also need your Reddit `username` and `password`.

2.  Create the `.env` File:
       In the project's root directory (the same folder as `agent.py`), create a new file and name it "exactly" `.env` (a period followed by "env", with no other extension like `.txt`).
       If you're on Windows and have trouble creating a file starting with a period, you can create `env.txt` first, then rename it to `.env` in the command prompt: `ren env.txt .env` (make sure file extensions are visible in File Explorer).

3.  Add Your Keys to the `.env` File:**
    Open the `.env` file with a text editor (like Notepad or VS Code) and add your credentials in the following format, replacing the placeholder text with your actual keys:
    ```env
    GITHUB_TOKEN=your_github_personal_access_token_here
    REDDIT_CLIENT_ID=your_reddit_app_client_id_here
    REDDIT_CLIENT_SECRET=your_reddit_app_client_secret_here
    REDDIT_USERNAME=your_reddit_username_here
    REDDIT_PASSWORD=your_reddit_password_here
    ```
       Save the `.env` file.
      IMPORTANT: This `.env` file contains sensitive information and should **never** be shared publicly or committed to a version control system like Git. If the project uses Git, make sure `.env` is listed in a `.gitignore` file.

4.    Reddit User Agent (in `agent.py`):
       Open the `agent.py` file.
      Find the section where `praw.Reddit` is initialized.
       Modify the `user_agent` string to be unique and descriptive, including your Reddit username. This helps Reddit identify your script.
        ```python
        # Example:
        reddit = praw.Reddit(
            # ... other parameters ...
            user_agent="InsightAgentProject by u/YourRedditUsername" # CHANGE YourRedditUsername
        )
        ```


Key changes in this version:

Step 1 (Setup): Assumes files are already present, focuses on Python installation.

Step 2 (Setup): Emphasizes navigating to the correct project directory.

Step 4 (Setup): Provides instructions for both using requirements.txt and installing packages manually if requirements.txt is missing. Includes a note for the project owner to create requirements.txt.

Configuration Section: More detailed steps on how to obtain the API keys for someone who might be doing it for the first time.

Configuration Section (Step 2): More explicit instructions on creating the .env file, including a tip for Windows users.

Configuration Section (Step 4): Highlights the need to update the user_agent in agent.py.

This should be clearer for someone who receives the project files directly without cloning from a remote repository. Remember to include a requirements.txt file in your project for the smoothest setup!

    🚀 Usage (Command-Line Interface)

All commands are run from the project's root directory using `python agent.py <command> [options...]`.
 1. Fetching Data

*   **Fetch data from GitHub:**
    ```bash
    python agent.py fetch github "<search_query>" [limit]
    ```
    *   `"<search_query>"`: The term to search for on GitHub (e.g., "large language models").
    *   `[limit]` (optional): Number of repositories to fetch (default is 10).
    *   Example: `python agent.py fetch github "AI ethics tools" 15`
    *   Data is saved to a JSON file in the `data/` directory (e.g., `data/github_AI_ethics_tools_YYYYMMDDHHMMSS.json`).

*   **Fetch data from Reddit:**
    ```bash
    python agent.py fetch reddit <subreddit_name> "<search_query>" [limit]
    ```
    *   `<subreddit_name>`: The name of the subreddit (e.g., `MachineLearning`).
    *   `"<search_query>"`: The term to search for within the subreddit.
    *   `[limit]` (optional): Number of posts to fetch (default is 10).
    *   Example: `python agent.py fetch reddit datascience "career advice" 5`
    *   Data is saved to a JSON file in the `data/` directory.

 2. Analyzing Data

*   **Perform analysis on fetched data:**
    ```bash
    python agent.py analyze <data_file_path> [library_name_to_search]
    ```
       `<data_file_path>`: Path to the JSON file generated by the `fetch` command (e.g., `data/github_AI_ethics_tools_YYYYMMDDHHMMSS.json`).
       `[library_name_to_search]` (optional): A specific library or keyword to count mentions for (e.g., "transformers", "pytorch").
       Example (general analysis): `python agent.py analyze data/github_AI_ethics_tools_YYYYMMDDHHMMSS.json`
       Example (library search): `python agent.py analyze data/github_AI_ethics_tools_YYYYMMDDHHMMSS.json "tensorflow"`

 3. Asking the AI a Question

   Ask the AI a question based on the context from a fetched data file:**
    ```bash
    python agent.py ask "<question_in_quotes>" <data_file_path> [index_of_entry]
    ```
       `"<question_in_quotes>"`: The question you want to ask the AI.
       `<data_file_path>`: Path to the JSON file containing the context (e.g., a GitHub data file).
       `[index_of_entry]` (optional): The index of the specific item (repository/post) in the JSON file to use as context (0 for the first item, 1 for the second, etc.). Defaults to 0 if not provided.
       Example: `python agent.py ask "What is the main purpose of this repository?" data/github_AI_ethics_tools_YYYYMMDDHHMMSS.json 0`
       Example: `python agent.py ask "What challenges are discussed in this post?" data/reddit_datascience_career_advice_YYYYMMDDHHMMSS.json 1`

    *Note: The first time you run an `ask` command, the Hugging Face QA model (`distilbert-base-cased-distilled-squad`) will be downloaded. This might take a few minutes.*

 Technologies Used::

  Python 3
  Hugging Face Transformers: For accessing and using pre-trained LLMs (specifically for Question Answering).
  PyTorch: As a backend for Hugging Face Transformers.
  Pandas: For data manipulation and analysis.
  PyGithub: Python library to access the GitHub API v3.
  PRAW (Python Reddit API Wrapper): Python library to access the Reddit API.
  python-dotenv: For managing environment variables from a `.env` file.


If you encounter any issues, please ensure your API keys are correct, dependencies are installed, and the commands are entered as specified.


