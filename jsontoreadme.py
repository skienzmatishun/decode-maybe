import json

def json_to_markdown(json_data):
    markdown = ""

    # Project Title
    markdown += f"# {json_data['project']['title']}\n\n"
    
    # Project Description
    markdown += "## Project Description\n"
    markdown += json_data['project']['description'].replace('\n', ' ') + "\n\n"
    
    # Project Goals
    markdown += "## Goals\n"
    for goal in json_data['project']['goals']:
        markdown += f"- {goal}\n"
    markdown += "\n"

    # Setup
    markdown += "## Setup\n"
    markdown += "### Dependencies\n"
    markdown += ", ".join(json_data['setup']['dependencies']) + "\n\n"
    markdown += "### Installation\n"
    markdown += json_data['setup']['installation']['instructions'] + "\n\n"

    # Usage
    markdown += "## Usage\n"
    markdown += "### Workflow\n"
    for step in json_data['usage']['workflow']:
        markdown += f"- {step}\n"
    markdown += "\n"
    markdown += "### Directories\n"
    markdown += f"**Input:** {json_data['usage']['directories']['input']}\n"
    markdown += f"**Output:** {', '.join(json_data['usage']['directories']['output'])}\n\n"

    # Files
    markdown += "## Key Files\n"
    markdown += "### Analysis Scripts\n"
    markdown += "| File | Purpose |\n"
    markdown += "|------|---------|\n"
    for script in json_data['files']['analysis_scripts']:
        markdown += f"| `{script['name']}` | {script['purpose']} |\n"
    markdown += "\n### Helper Scripts\n"
    markdown += "| File | Purpose |\n"
    markdown += "|------|---------|\n"
    for helper in json_data['files']['helper_scripts']:
        name, purpose = helper.split(" (")
        markdown += f"| `{name}` | {purpose.rstrip(')')} |\n"
    markdown += "\n"

    # Validation
    markdown += "## Validation\n"
    markdown += "### Metrics\n"
    markdown += ", ".join(json_data['validation']['metrics']) + "\n\n"
    markdown += "### Success Criteria\n"
    markdown += json_data['validation']['success_criteria'] + "\n\n"

    # Contributors
    markdown += "## Contributors\n"
    for contributor in json_data['contributors']:
        markdown += f"- **{contributor['name']}**: {contributor['role']} ({contributor['contact']})\n"
    markdown += "\n"

    # License
    markdown += "## License\n"
    markdown += json_data['license'] + "\n\n"

    # Acknowledgments
    markdown += "## Acknowledgments\n"
    for ack in json_data['acknowledgments']:
        markdown += f"- {ack}\n"

    return markdown

if __name__ == "__main__":
    # Load JSON data
    with open("readme.json", "r") as json_file:
        data = json.load(json_file)
    
    # Convert to markdown
    md_content = json_to_markdown(data)
    
    # Save to file
    with open("README.md", "w") as md_file:
        md_file.write(md_content)
    
    print("Markdown documentation generated successfully as README.md")