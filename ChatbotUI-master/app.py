from flask import Flask, render_template, request
import pandas as pd
import os
from together import Together
import json
import re

app = Flask(__name__)


# toaster's together key
os.environ["TOGETHER_API_KEY"] = "f00e4f3943a505ce68ff471fb2976c33040fcbb6c16874121ad76dbc53b70952" 
together = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# currently a list of all ingredients that appears in our recipe list
# in future development, this should be directly retrieved from supermarkets' available product list
ingredient_df = pd.read_csv('supermarket_product_list.csv', index_col=0)
ingredient_list = ingredient_df.index.to_list()
discounted_ingredients = set(ingredient_df[ingredient_df['discount'].notnull()].index.to_list())
# Naive implementation: only record the number of ingredients that are currently discounted
recipe_df = pd.read_csv('data_recipe_unnumbered.csv')
recipe_df['discount_count'] = recipe_df['ingredient_en'].apply(lambda x: len(set(x.split(', ')).intersection(set(discounted_ingredients))))

with open("functions.json", "r") as json_file:
    funcs = json.load(json_file)
funcs[0]["function"]["parameters"]["properties"]["include_ingredient"]["items"]["enum"] = ingredient_list
# funcs[0]["function"]["parameters"]["properties"]["exclude_ingredient"]["items"]["enum"] = ingredient_list

messages = [
    {"role": "system", 
     "content": 
     "You are a helpful cooking assistant whose main purpose is to recommend recipes. "
     "You can provide suitable recipes according to what users want to include and exclude. "
     "Please provide responses based on the users' requests. "
     "If you are unable to understand the user's request, you should ask for clarification. "}
]
    
# Notable issue: When user asks to exclude an item not in our built in list of ingredients, AI panics and excludes everything that isn't included. More testing required.

# main feature
def recommend_recipes(include_ingredient: list[str], 
                      exclude_ingredient: list[str], 
                      recipe_count: int=1,
                      exclude_id: list[int]=[]) -> str:
    global recipe_df
    print("Include: ", include_ingredient)
    print("Exclude: ", exclude_ingredient)
    print("Recipe_count: ", recipe_count)
    remaining_recipes = recipe_df
    # first we remove recipes containing excluded ingredients
    for ingredient in exclude_ingredient:
        # print(~remaining_recipes['ingredient_en'].str.contains(', '+ingredient+',', case=False, na=False))
        remaining_recipes = remaining_recipes[~remaining_recipes['ingredient_en'].str.contains(ingredient, case=False, na=False)]
        print("remaining_recipes: \n", 
              remaining_recipes[~remaining_recipes['ingredient_en'].str.contains(', '+ingredient+',', case=False, na=False)])
    include_set = set([i.lower() for i in include_ingredient])
    ans = []
    # Naive implementation: repeatedly select the recipe that ticks off the most items in the include list
    for i in range(max(recipe_count, 1)):
        print(f'Dish {i}:') 
        remaining_recipes['matches'] = remaining_recipes['ingredient_en'].apply(lambda x: len(set(x.split(', ')).intersection(set(include_set))))
        remaining_recipes = remaining_recipes.sort_values(by=["matches", "discount_count"], ascending=False)
        print("remaining_recipes: \n", remaining_recipes)
        head = remaining_recipes.head(1)
        print("Head: ", head)
        if head.empty:  # no recipe found
            break
        remaining_recipes = remaining_recipes.drop(head.index[0])
        selected_recipe = head.T[head.index[0]].to_list()
        selected_recipe.append(head.index[0])
        ans.append(selected_recipe)
        print(selected_recipe)
        include_set = include_set - set([ing.lower() for ing in ans[-1][1].split(', ')])
    return_recipes = ""
    for i in range(len(ans)):
        return_recipes += f"Recipe {i+1}: {ans[i][0]} (recipe_id {ans[i][3]})\nIngredients: {ans[i][1]}\nInstructions: \n{ans[i][2]}\n\n"
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
                    include_ingredient=function_args.get("include_ingredient"),
                    exclude_ingredient=function_args.get("exclude_ingredient"),
                    recipe_count=function_args.get("recipe_count"),
                    exclude_id=function_args.get("exclude_id"),
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
    if response_content == "":
        response_content = "Sorry, I don't understand. Please provide more information."
    # return final content string
    return response_content


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=54321)