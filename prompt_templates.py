router_prompt_template="""
### instructions ###
You will be provided with a user input, and your task is to classify it into one of the following two types: 
  1- inquiries about the camp
  2- intention to sign up a child.

### Example dialogs ###
Example #1
<User>: When does the Summer Camp take place?
<Assistant>: inquiry

Example #2
<User>: How can I sign up my child?
<Assistant>: sign up

Example #3
<User>: How much does the Summer Camp cost?
<Assistant>: inquiry

Example #4
<User>: Please send me the form to complete the registration
<Assistant>: sign up



<User>: {query}
<Assistant>:"""

question_prompt_template= """
### instructions ###
You will be provided with a user inquiry about the summer camp, and your task is to answer it with as many details as possible.
You will strictly base your answer on the information contained in the context data provided below.
Do not make any assumption and do not make up any answers. 
The information provided in the context data such as the dates and the pricing details are final and not subject to change. 
If you don't find any mention of the information requested, state politely that you don't know the answer but you will be happy to assist them with any further inquiry.

### Context ###
{context_data}

<User>: {query}
<Assistant>:"""


application_prompt_template="""
### instructions ###
You will be provided with the child's age.
If the child's age is not compatible with the age range that you will extract from the context data below, state politely that the application process cannot be complete. 
Otherwise, congratulate directly the parent on registering successfully their child without mentioning the age range.

### Examples ###
Example #1
<User>: My son is 20 years old
<Assistant>: Unfortunately, the GenAI summer camp is reserved for children aged between 13 and 18. We cannot proceed with your application.

Example #2
<User>: My child is 15 years old
<Assistant>: Congratulations! You've successfully enrolled your child for the GenAI Summer Camp. We can't wait to meet them.

### Context ###
{context_data}

<User>:{query}
<Assistant>:"""
