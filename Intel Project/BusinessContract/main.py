 

#pip install PyPDF2
#pip install transformers
#pip install torch
#pip install PyMuPDF
import fitz 

import PyPDF2
from transformers import pipeline

# Load the PDF file
pdf_file_path = 'BusinessContract\BusinessContract2.pdf'  
with open(pdf_file_path, 'rb') as pdf_file:
  pdf_reader = PyPDF2.PdfReader(pdf_file)

pdf_path = 'BusinessContract\BusinessContract2.pdf'



try:
    # Open the PDF file in binary mode
    with open(pdf_path, 'rb') as pdf_file:
        # Initialize the PDF reader
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Initialize a variable to hold the extracted text
        text = ""

        # Iterate through all the pages and extract text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        # Print the extracted text
        print(text)

except FileNotFoundError:
    print(f"Error: The file '{pdf_path}' was not found.")
except PyPDF2.utils.PdfReadError as e:
    print(f"Error reading PDF: {e}")



# Load a pre-trained NLP model for classification
classifier = pipeline("zero-shot-classification")

# Define candidate labels for content and clause classification
content_labels = ["NDA", "Sales Agreement", "Employment Contract", "Lease Agreement"]
clause_labels = ["Confidentiality", "Payment Terms", "Termination", "Liability"]

# Classify the content of the contract
# Load the PDF file
pdf_file_path = '/content/BusinessContract2.pdf' 

try:
    # Open the PDF file in binary mode
    with open(pdf_file_path, 'rb') as pdf_file:
        # Initialize the PDF reader
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)

        # Initialize a variable to hold the extracted text
        text = ""

        # Iterate through all the pages and extract text
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extract_text()

        # Print the extracted text
        print("Extracted Text:", text[:1000])  # Displaying the first 1000 characters for brevity

except FileNotFoundError:
    print(f"The file {pdf_file_path} was not found.")
    text = ""
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    text = ""

# Load a pre-trained NLP model for classification
classifier = pipeline("zero-shot-classification")

# Define candidate labels for content classification
content_labels = ["NDA", "Sales Agreement", "Employment Contract", "Lease Agreement"]

# Ensure text is not empty before classifying
if text.strip() and content_labels:
    # Classify the content of the contract
    content_result = classifier(text, content_labels)
    print("Content Classification:")
    for label, score in zip(content_result["labels"], content_result["scores"]):
        print(f"{label}: {score:.4f}")
else:
    print("No valid text or labels to classify.")

# Split the contract into clauses (you'll need a more sophisticated method for real-world contracts)
clauses = text.split('\n\n')  

# Classify each clause
for clause in clauses:
    if clause.strip() and clause_labels:
        clause_result = classifier(clause, clause_labels)
        print("\nClause Classification:")
        for label, score in zip(clause_result["labels"], clause_result["scores"]):
            print(f"{label}: {score:.4f}")
    else:
        print("Empty clause or no labels to classify.")

# Deviation detection (requires a template contract for comparison)
# This is a simplified example, you'll need a more robust approach for real-world scenarios
template_text = """
SOCIAL MEDIA MANAGEMENT CONTRACTUAL AGREEMENT

PARTIES
-This Social Media Management Contractual Agreement (hereinafter referred to as the "Agreement") is entered into on 12/12/9, by and between Digital Dreams, with an address of Andheri ,Mumbai (hereinafter referred to as the "Client") and John Smith, with an address of 123 Social Media Street, Suite 101, Social Media City, SM 12345, United States(hereinafter referred to as the "Social Media Manager") (collectively referred to as the "Parties").



SERVICES PROVIDED
The Social Media Manager agrees to provide comprehensive social media management services, tailored to the Client's specific needs and objectives. These services include but are not limited to:
1. Development of a customized social media strategy aligned with the Client's brand identity and marketing goals.
2. Creation of engaging and relevant content for distribution across various social media platforms, including but not limited to Facebook, Instagram, Twitter, LinkedIn, and TikTok.
3. Regular posting and scheduling of content to maximize audience engagement and reach.
4. Monitoring and moderation of social media channels to ensure timely responses to comments, messages, and inquiries.
5. Implementation of targeted advertising campaigns to increase brand visibility, drive website traffic,

"""

template_clauses = template_text.split('\n\n')

for i, (clause, template_clause) in enumerate(zip(clauses, template_clauses)):
  if clause != template_clause:
    print(f"\nDeviation detected in clause {i+1}")
