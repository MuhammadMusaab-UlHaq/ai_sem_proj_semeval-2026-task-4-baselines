import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# --- 1. Setup ---
# NLTK package 'punkt' should be downloaded from the previous step.

# Define file paths based on project plan
DATA_FILE = 'data/dev_track_a.jsonl' 
PLOT_DIR = 'plots/'

# --- 2. Load Data ---
# Task: Load the data
try:
    df = pd.read_json(DATA_FILE, lines=True)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please make sure 'dev_track_a.jsonl' exists in the 'data' folder.")
    exit()
except ValueError as e:
    print(f"Error reading {DATA_FILE}. Is the file empty or malformed?")
    print(f"Details: {e}")
    exit()

print("Data loaded successfully:")
print(df.head())

# --- 3. Analyze Story Lengths (Tokens & Sentences) ---
# Task: Distribution of story lengths (tokens and sentences)

# Helper functions to count tokens and sentences
def count_tokens(text):
    if pd.isna(text):
        return 0
    return len(word_tokenize(str(text)))

def count_sentences(text):
    if pd.isna(text):
        return 0
    return len(sent_tokenize(str(text)))

# Apply the functions to all text columns
text_columns = ['anchor_text', 'text_a', 'text_b']
for col in text_columns:
    df[f'{col}_token_count'] = df[col].apply(count_tokens)
    df[f'{col}_sentence_count'] = df[col].apply(count_sentences)

print("\nData with counts:")
# *** FIX: Corrected 'anchor_token_count' to 'anchor_text_token_count' ***
print(df[['anchor_text_token_count', 'text_a_token_count', 'text_b_token_count']].describe())

# --- 4. Generate Plots ---
# Task: Generate plots for story lengths and A/B labels

# Plot 1: Distribution of Story Lengths (Tokens)
print("\nGenerating token distribution plot...")
plt.figure(figsize=(12, 7))
# *** FIX: Corrected 'anchor_token_count' to 'anchor_text_token_count' ***
sns.histplot(df['anchor_text_token_count'], color='blue', label='Anchor Text', kde=True, stat="density", element="step")
sns.histplot(df['text_a_token_count'], color='green', label='Text A', kde=True, stat="density", element="step")
sns.histplot(df['text_b_token_count'], color='red', label='Text B', kde=True, stat="density", element="step")
plt.title('Distribution of Story Lengths (in Tokens)')
plt.xlabel('Token Count')
plt.ylabel('Density')
plt.legend()
plt.savefig(f'{PLOT_DIR}story_length_tokens_dist.png')
print(f"Saved token plot to {PLOT_DIR}story_length_tokens_dist.png")

# Plot 2: Distribution of Story Lengths (Sentences)
print("Generating sentence distribution plot...")
plt.figure(figsize=(12, 7))
# *** FIX: Corrected 'anchor_sentence_count' to 'anchor_text_sentence_count' ***
sns.histplot(df['anchor_text_sentence_count'], color='blue', label='Anchor Text', kde=False, stat="count", element="step", binwidth=1)
sns.histplot(df['text_a_sentence_count'], color='green', label='Text A', kde=False, stat="count", element="step", binwidth=1)
sns.histplot(df['text_b_sentence_count'], color='red', label='Text B', kde=False, stat="count", element="step", binwidth=1)
plt.title('Distribution of Story Lengths (in Sentences)')
plt.xlabel('Sentence Count')
plt.ylabel('Frequency (Count)')
plt.legend()
# *** FIX: Corrected 'anchor_sentence_count' to 'anchor_text_sentence_count' ***
plt.xlim(0, df['anchor_text_sentence_count'].max() + 2) # Adjust x-axis limit
plt.savefig(f'{PLOT_DIR}story_length_sentences_dist.png')
print(f"Saved sentence plot to {PLOT_DIR}story_length_sentences_dist.png")


# Plot 3: Balance of A/B Labels
# Task: Generate plot for the balance of the A/B labels
print("Generating A/B label balance plot...")
plt.figure(figsize=(8, 6))
sns.countplot(x='text_a_is_closer', data=df)
plt.title('Balance of A/B Labels ("text_a_is_closer")')
plt.xlabel('Is Text A Closer?')
plt.ylabel('Count')
plt.savefig(f'{PLOT_DIR}ab_label_balance.png')
print(f"Saved A/B label balance plot to {PLOT_DIR}ab_label_balance.png")

print("\nAnalysis complete.")