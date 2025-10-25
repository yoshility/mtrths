from prm import prm

step = [
    {"role": "system", "content": "You are a helpful assistant solving math problems."},
    {"role": "user", "content": "Five friends eat at a fast-food chain and order the following: 5 pieces of hamburger that cost $3 each; 4 sets of French fries that cost $1.20; 5 cups of soda that cost $0.5 each; and 1 platter of spaghetti that cost $2.7. How much will each of them pay if they will split the bill equally?"},
    {"role": "assistant", "content": "To find the total cost of the order, we need to calculate the cost of each item and add them together. \n\n- 5 pieces of hamburger at $3 each will cost 5 x 3 = $15.\n- 4 sets of French fries at $1.20 each will cost 4 x 1.20 = $4.80.\n- 5 cups of soda at $0.5 each will cost 5 x 0.5 = $2.50.\n- The platter of spaghetti costs $2.7.\n\nNow, let's add these costs together: \n$15 + $4.80 + $2.50 + $2.7 = $25\n\nSince there are 5 friends, we need to divide the total cost by 5 to find out how much each friend will pay: \n$25 ÷ 5 = $5\n\nEach friend will pay $5."}
]

print(prm(step))