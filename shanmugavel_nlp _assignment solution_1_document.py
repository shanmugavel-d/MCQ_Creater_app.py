# -*- coding: utf-8 -*-
"""nlp_assignment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14evNyV-6qocxneNZSjoc1CZ2p8sHsSZn
"""

pip install openai

"""Description:

This code generates multiple-choice questions from a given context using OpenAI GPT-3.5. The code first validates the input and then generates the questions. The questions are then formatted and printed.

Usage:

To use the code, you will need to have an OpenAI API key. You can get an API key from the OpenAI website. Once you have an API key, you can run the code by providing the context and API key as arguments. For example, to generate questions from the context "This is a test", you would run the code like this:

Python
get_mca_questions("This is a test", "YOUR_API_KEY")
Use code with caution. Learn more
Output:

The output of the code will be a list of formatted questions and options. For example, if the context is "This is a test", the output might be:

Q1: What is this?
A. A test.
B. A sentence.
C. A paragraph.
D. A document.

Q2: What is the purpose of this text?
A. To test the code.
B. To provide an example of a multiple-choice question.
C. To demonstrate the use of OpenAI GPT-3.5.
D. All of the above.
Notes:

The code uses the openai library to interact with the OpenAI API.
The code is designed to be run from a Python interpreter

"""

# import openai

# # Set your OpenAI API key here
# openai.api_key = "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"

# def get_mca_questions(context: str):
#     # Validate input
#     if not isinstance(context, str):
#         raise TypeError("Input context must be a string.")

#     # Generate multiple-choice questions
#     mca_questions = []

#     prompt = f"Generate multiple-choice questions for the following passage:\n{context}"
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         max_tokens=150,  # Set the desired length of the generated questions
#         n=10,  # Number of questions to generate
#         stop=None,  # Set a stop token if necessary to control question length
#     )

#     # Extract the generated questions from the API response
#     for choice in response["choices"]:
#         mca_questions.append(choice["text"].strip())

#     return mca_questions

context = input()

#a = get_mca_questions(context)

#print(a)

import openai

def get_mca_questions(context: str, api_key: str):
    # Validate input
    if not isinstance(context, str):
        raise TypeError("Input context must be a string.")

    # Validate API key
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("Invalid API key provided.")

    # Step 1: Preprocess the text (optional)
    # You can preprocess the text here if required.

    # Step 2: Generate questions using OpenAI GPT-3.5
    mca_questions = []

    prompt = f"Generate only 3 multiple-choice questions which have multiple correct answers for the following passage:\n{context}"
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150,  # Set the desired length of the generated questions
            n=10,  # Number of questions to generate
            stop=None,  # Set a stop token if necessary to control question length
            api_key=api_key,  # Use the provided API key
        )

        # Extract the generated questions from the API response
        for choice in response["choices"]:
            mca_questions.append(choice["text"].strip())

    except Exception as e:
        # Handle API errors
        raise Exception(f"Error generating questions: {str(e)}")

    return mca_questions

api_key = "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"

get_mca_questions(context,api_key)[:2]

def format_question_options(input_list):
    questions = []
    options = []

    for entry in input_list:
        q_and_a = entry.split('\n\n')
        question = q_and_a[0].strip()
        options_and_answers = q_and_a[0].split('\n')

        question_number = question[0]  # Assuming the question always starts with a number followed by a period.
        formatted_question = f"Q{question_number}: {question[3:]}"  # Removing the question number and period.

        questions.append(formatted_question)

        question_options = [option[3:] for option in options_and_answers[:-1]]
        options.append(question_options)

    return questions, options

# Given list of questions and options
given_list = get_mca_questions(context,api_key)

questions, options = format_question_options(given_list)

# Printing the formatted questions and options
for i, question in enumerate(questions):
    print(question)
    for j, option in enumerate(options[i]):
        print(f"{chr(j + 97)}. {option}")
    print()

