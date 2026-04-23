def process(data):
    total = 0
    for item in data:
        total =+ item
    return total / len(data)

print(process([10, 20, 30]))
print(process([]))