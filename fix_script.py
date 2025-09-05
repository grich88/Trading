with open('app_enhanced_visual.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the line break in current_rsi
fixed_content = content.replace('current_\nrsi', 'current_rsi')

with open('app_enhanced_visual.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("File fixed successfully!")
