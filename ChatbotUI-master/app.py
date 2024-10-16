from flask import Flask, render_template, request
import pandas as pd
import os
from together import Together
import json
import re

app = Flask(__name__)



recipe_df = pd.read_csv('data_recipe_unnumbered.csv')
# print("recipe_df content: \n", recipe_df)

# toaster's together key
os.environ["TOGETHER_API_KEY"] = "f00e4f3943a505ce68ff471fb2976c33040fcbb6c16874121ad76dbc53b70952" 

# ideally, we build this list by checking all ingredients that appear in our recipe dataset
ingredient_list = [
    'Beef', 'Green onion', 'Garlic', 'Ginger', 'Oyster sauce', 'Sugar', 'Cornstarch', 'Oil', 'Salt', 'Soy sauce', 
    'Black pepper powder', 'Eggs', 'Dried whitebait', 'Chicken wings', 'Potatoes', 'Onion', 'Dark soy sauce', 
    'Light soy sauce', 'Pepper', 'Sesame oil', 'Spring chicken', 'Carrot', 'Broccoli', 'Monk fruit sugar', 
    'Rice wine', 'Curry powder', 'Evaporated milk', 'Minced meat', 'Cold rice', 'Dried scallops', 
    'Choy sum stems', 'Egg whites', 'Coriander', 'Seasoning', 'Rice wine', 'Chicken fillet', 'Salted fish', 
    'Rice', 'Water spinach', 'Beef slices', 'Minced garlic', 'Chicken powder', 'Chili', 'Chicken drumsticks', 
    'Star anise', 'Cotton tofu', 'Braising sauce', 'Palm sugar or brown sugar'
]

together = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

with open("functions.json", "r") as json_file:
    funcs = json.load(json_file)

funcs[0]["function"]["parameters"]["properties"]["include"]["items"]["enum"] = ingredient_list
# funcs[0]["function"]["parameters"]["properties"]["exclude"]["items"]["enum"] = ingredient_list

messages = [
    {"role": "system", 
     "content": 
     "You are a helpful assistant that can access external functions. "
     "The responses from these function calls will be appended to this dialogue. "
     "Please provide responses based on the information from these function calls."}
]
    
# Notable issue: When user asks to exclude an item not in our built in list of ingredients, AI panics and excludes everything that isn't included. More testing required.

# main feature
def recommend_recipes(include: list[str], exclude: list[str], headcount: int=1):
    global recipe_df
    print("Include: ", include)
    print("Exclude: ", exclude)
    print("Headcount: ", headcount)
    remaining_recipes = recipe_df
    # first we remove recipes containing excluded ingredients
    for ingredient in exclude:
        # print(~remaining_recipes['ingredient_en'].str.contains(', '+ingredient+',', case=False, na=False))
        remaining_recipes = remaining_recipes[~remaining_recipes['ingredient_en'].str.contains(', '+ingredient+',', case=False, na=False)]
        print("remaining_recipes: \n", 
              remaining_recipes[~remaining_recipes['ingredient_en'].str.contains(', '+ingredient+',', case=False, na=False)])
    include_set = set([i.lower() for i in include])
    ans = []
    # Naive implementation: repeatedly pick the recipe that ticks off the most ingredients in the list
    for i in range(max(headcount-1, 1)):
        print(f'Dish {i}:') 
        remaining_recipes['matches'] = remaining_recipes['ingredient_en'].apply(lambda x: len(set(x.split(', ')).intersection(set(include_set))))
        remaining_recipes = remaining_recipes.sort_values(by="matches", ascending=False)
        print("remaining_recipes: \n", remaining_recipes)
        head = remaining_recipes.head(1)
        print("Head: ", head)
        remaining_recipes = remaining_recipes.drop(head.index[0])
        ans.append(head.T[head.index[0]].to_list())
        print(ans[-1])
        include_set = include_set - set([ing.lower() for ing in ans[-1][1].split(', ')])
    return_recipes = ""
    for i in range(len(ans)):
        return_recipes += f"Recipe {i+1}: {ans[i][0]}\nIngredients: {ans[i][1]}\nInstructions: \n{ans[i][2]}\n\n"
    return return_recipes




@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    global messages
    query = request.args.get('msg')

    messages.append({"role": "user", "content": query})
    response = together.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=messages,
        tools=funcs,
        tool_choice="auto",
    )
    tool_calls = response.choices[0].message.tool_calls
    while tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "recommend_recipes":
                function_response = recommend_recipes(
                    include=function_args.get("include"),
                    exclude=function_args.get("exclude"),
                    headcount=function_args.get("headcount"),
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
        response = together.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
        )
        tool_calls = response.choices[0].message.tool_calls
        print(json.dumps(response.choices[0].message.model_dump(), indent=2))

    # All tool calls are completed
    response_content = response.choices[0].message.content
    print(response_content)
    # there may need to be some extra bug handling here if any exists
    # return final content string
    return response_content


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=54321)