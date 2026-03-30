import os

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content.replace('input_dim=250', 'input_dim=1000')
    new_content = new_content.replace('num_classes=2', 'num_classes=5')
    new_content = new_content.replace('seq_len=250', 'seq_len=1000')
    new_content = new_content.replace('in_channels=1,', 'in_channels=12,')
    new_content = new_content.replace('in_channels=1)', 'in_channels=12)')
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed {filepath}")

for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))
