def run_quiz():
    questions = [
        {
            "question": "What is the capital of France?",
            "options": ["A) Berlin", "B) Madrid", "C) Paris", "D) Rome"],
            "answer": "C) Paris"
        },
        {
            "question": "What is 2 + 2?",
            "options": ["A) 3", "B) 4", "C) 5", "D) 6"],
            "answer": "B) 4"
        },
        {
            "question": "What is the largest planet in our solar system?",
            "options": ["A) Earth", "B) Jupiter", "C) Saturn", "D) Mars"],
            "answer": "B) Jupiter"
        },
        {
            "question": "Who wrote 'Hamlet'?",
            "options": ["A) Charles Dickens", "B) Mark Twain", "C) William Shakespeare", "D) Jane Austen"],
            "answer": "C) William Shakespeare"
        }
    ]

    score = 0
    '''
    enumerate function is used to get both index and value from the list. Example: 
    for index, value in enumerate(['a', 'b', 'c']):
        print(index, value)
    The output will be:
    0 a
    '''
    for index,q in enumerate(questions): 
        # print(index, q)
        print(f"Q{index + 1}: {q['question']}")
        for option in q['options']:
            print(option)
        
        user_answer = input("Your answer(A/B/C/D): ")
        # print(user_answer.strip().upper(), q['answer'][0])
        if user_answer.strip().upper() == q['answer'][0]:
            print("Correct!\n")
            score += 1

    print(f"Your final score is {score}/{len(questions)}")

run_quiz()