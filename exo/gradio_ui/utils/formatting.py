"""
Formatting utilities for the Gradio UI.
"""

import re
import html
from typing import Dict, List, Any, Optional

import markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes value into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string
    """
    if bytes_value == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while bytes_value >= 1024 and i < len(units) - 1:
        bytes_value /= 1024
        i += 1
    
    return f"{bytes_value:.2f} {units[i]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string
    """
    if seconds is None or seconds == 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from markdown text.
    
    Args:
        text: Markdown text
        
    Returns:
        List of dictionaries with language and code
    """
    pattern = r'```(\w*)\n([\s\S]*?)```'
    matches = re.findall(pattern, text)
    
    code_blocks = []
    for lang, code in matches:
        code_blocks.append({
            'language': lang.strip() or 'text',
            'code': code
        })
    
    return code_blocks


def process_think_blocks(text: str) -> str:
    """
    Process <think> blocks in text for displaying thought process.
    
    Args:
        text: Message text
        
    Returns:
        Processed HTML
    """
    pattern = r'<think>([\s\S]*?)(?:</think>|$)'
    
    def replace_think(match):
        content = match.group(1)
        complete = '</think>' in match.group(0)
        
        # Convert markdown in the thinking content
        formatted_content = render_markdown(content)
        
        # Create the HTML for the thinking block
        spinner = '' if complete else '<div class="thinking-spinner"></div>'
        return f'''
        <div class="thinking-block">
            <div class="thinking-header">
                Thinking{spinner}
            </div>
            <div class="thinking-content">
                {formatted_content}
            </div>
        </div>
        '''
    
    return re.sub(pattern, replace_think, text)


def render_code_block(code: str, language: str = '') -> str:
    """
    Render a code block with syntax highlighting.
    
    Args:
        code: Code content
        language: Programming language
        
    Returns:
        HTML formatted code
    """
    try:
        if language and language != 'text':
            lexer = get_lexer_by_name(language, stripall=True)
        else:
            lexer = guess_lexer(code)
    except Exception:
        # If language detection fails, use plain text
        from pygments.lexers.special import TextLexer
        lexer = TextLexer()
    
    formatter = HtmlFormatter(
        style='monokai',
        cssclass='codeblock',
        linenos=False,
        wrapcode=True
    )
    
    highlighted = highlight(code, lexer, formatter)
    
    # Add copy button
    return f'''
    <div class="code-block-wrapper">
        {highlighted}
        <button class="copy-code-button" onclick="copyCode(this)">
            <span class="copy-icon">📋</span>
        </button>
    </div>
    <script>
    function copyCode(button) {{
        const codeBlock = button.parentElement.querySelector('pre');
        const code = codeBlock.textContent;
        
        navigator.clipboard.writeText(code).then(() => {{
            const icon = button.querySelector('.copy-icon');
            const originalText = icon.textContent;
            icon.textContent = '✓';
            setTimeout(() => {{
                icon.textContent = originalText;
            }}, 2000);
        }}).catch(err => {{
            console.error('Failed to copy code: ', err);
        }});
    }}
    </script>
    '''


def render_markdown(text: str) -> str:
    """
    Render markdown text to HTML with code highlighting.
    
    Args:
        text: Markdown text
        
    Returns:
        HTML formatted text
    """
    # First process <think> blocks
    text = process_think_blocks(text)
    
    # Extract code blocks to process them separately
    code_blocks = extract_code_blocks(text)
    
    # Replace code blocks with placeholders
    for i, block in enumerate(code_blocks):
        placeholder = f"CODE_BLOCK_PLACEHOLDER_{i}"
        text = text.replace(f"```{block['language']}\n{block['code']}```", placeholder)
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        text,
        extensions=['extra', 'nl2br', 'sane_lists', 'smarty']
    )
    
    # Replace placeholders with highlighted code
    for i, block in enumerate(code_blocks):
        placeholder = f"CODE_BLOCK_PLACEHOLDER_{i}"
        highlighted_code = render_code_block(block['code'], block['language'])
        html_content = html_content.replace(f"<p>{placeholder}</p>", highlighted_code)
        html_content = html_content.replace(placeholder, highlighted_code)
    
    return html_content


def format_message_for_display(message: Dict[str, Any]) -> str:
    """
    Format a message for display in the chat UI.
    
    Args:
        message: Message dictionary with role and content
        
    Returns:
        HTML formatted message
    """
    role = message.get('role', 'assistant')
    content = message.get('content', '')
    
    # Handle image content
    if isinstance(content, list):
        text_parts = []
        image_urls = []
        
        for part in content:
            if part.get('type') == 'text':
                text_parts.append(part.get('text', ''))
            elif part.get('type') == 'image_url':
                image_url = part.get('image_url', {}).get('url', '')
                if image_url:
                    image_urls.append(image_url)
        
        # Combine text parts
        text_content = ' '.join(text_parts)
        
        # Add image previews at the top
        image_previews = ''
        for url in image_urls:
            image_previews += f'<div class="message-image"><img src="{html.escape(url)}" alt="Image" /></div>'
        
        # Render text as markdown
        formatted_text = render_markdown(text_content) if text_content else ''
        
        return f'{image_previews}{formatted_text}'
    else:
        # Simple text message
        return render_markdown(content)


def format_chat_history(history: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Format chat history from Gradio format to API format.
    
    Args:
        history: List of [user_message, assistant_message] pairs
        
    Returns:
        List of message dictionaries with role and content
    """
    messages = []
    
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    return messages