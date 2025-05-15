"""
Model selector component for Gradio UI.
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
import gradio as gr

from ..api import get_api_client


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


def group_models_by_prefix(models: Dict[str, Any], show_downloaded_only: bool = False) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Group models by prefix for hierarchical display.
    
    Args:
        models: Dictionary of models
        show_downloaded_only: Whether to show only downloaded models
        
    Returns:
        Nested dictionary of grouped models
    """
    # Create models structure
    groups = {
        'LOCAL': {
            'LLAMA': {},
            'MISTRAL': {},
            'DEEPSEEK': {},
            'PHI': {},
            'QWEN': {},
            'GEMMA2': {},
            'LLAVA': {},
            'STABLE': {},
            'NEMOTRON': {},
            'DUMMY': {}
        },
        'CLOUD': {
            'OPENAI': {},
            'ANTHROPIC': {}
        },
        'NETWORK': {}  # Network/peer models
    }
    
    # Process all models
    for model_id, model in models.items():
        if not model or not model_id:
            continue
        
        try:
            # Cloud models
            if (model.get('isCloudModel') or 
                model.get('provider') in ['openai', 'anthropic'] or 
                model_id.startswith('gpt-') or 
                model_id.startswith('claude-')):
                
                provider = (model.get('provider', '').upper() or 
                           ('OPENAI' if model_id.startswith('gpt-') else 
                            'ANTHROPIC' if model_id.startswith('claude-') else 'OTHER'))
                
                if provider not in groups['CLOUD']:
                    groups['CLOUD'][provider] = {}
                
                groups['CLOUD'][provider][model_id] = model
            
            # Network models
            elif model.get('isPeerModel') or model.get('peer_id') or model.get('node_id'):
                peer_name = model.get('peer_name') or model.get('node_name') or 'UNKNOWN'
                
                if peer_name not in groups['NETWORK']:
                    groups['NETWORK'][peer_name] = {}
                
                groups['NETWORK'][peer_name][model_id] = model
            
            # Local models
            else:
                # Parse model category from name
                category = 'OTHER'
                if '-' in model_id:
                    parts = model_id.split('-')
                    category = parts[0].upper()
                
                if category not in groups['LOCAL']:
                    groups['LOCAL'][category] = {}
                
                groups['LOCAL'][category][model_id] = model
        except Exception as e:
            print(f"Error processing model {model_id}: {e}")
    
    # Apply downloaded-only filter if enabled
    if show_downloaded_only:
        for group_type in groups:
            for category in list(groups[group_type].keys()):
                category_models = groups[group_type][category]
                filtered_models = {}
                
                for model_id, model in category_models.items():
                    # For cloud models, they're always "downloaded"
                    if model.get('provider') and model.get('provider') != 'exo':
                        filtered_models[model_id] = model
                    # For local models, check downloaded status
                    elif (model.get('downloaded') is True or 
                          model.get('ready') is True or 
                          model.get('download_percentage') == 100):
                        filtered_models[model_id] = model
                
                groups[group_type][category] = filtered_models
    
    # Clean up empty categories
    for group_type in list(groups.keys()):
        for category in list(groups[group_type].keys()):
            if (len(groups[group_type][category]) == 0 and
                not (group_type == 'LOCAL' and category in ['LLAMA', 'MISTRAL', 'DEEPSEEK']) and
                not (group_type == 'CLOUD' and category in ['OPENAI', 'ANTHROPIC'])):
                del groups[group_type][category]
    
    return groups


def create_model_selector(
    on_model_select: Callable[[str], None],
    on_download_model: Callable[[str], None] = None,
    on_delete_model: Callable[[str], None] = None
) -> Tuple[gr.Accordion, Callable[[], Dict[str, Any]], Callable[[str], None]]:
    """
    Create a model selector component.
    
    Args:
        on_model_select: Callback when a model is selected
        on_download_model: Callback when model download is requested
        on_delete_model: Callback when model deletion is requested
        
    Returns:
        Tuple of (accordion component, refresh function, select function)
    """
    api_client = get_api_client()
    model_state = gr.State({})
    selected_model = gr.State("")
    show_downloaded_only = gr.State(False)
    
    def fetch_models():
        """Fetch models from the API and return them."""
        try:
            models_data = api_client.get_models()
            initial_models = api_client.get_initial_models()
            
            # Process models data
            models = {}
            
            # Add initial models (local models)
            for model_id, model_data in initial_models.items():
                models[model_id] = {
                    **model_data,
                    'id': model_id,
                    'name': model_data.get('name') or model_data.get('display_name') or model_id,
                    'loading': False,
                    'provider': 'exo',
                    'isLocalModel': True,
                    'total_size': model_data.get('total_size') or 0,
                    'download_percentage': model_data.get('download_percentage') or 0,
                    'downloaded': model_data.get('downloaded') is True,
                    'canDelete': True
                }
            
            # Add from models API (includes cloud models)
            if models_data and 'data' in models_data and isinstance(models_data['data'], list):
                for model in models_data['data']:
                    if not model.get('id'):
                        continue
                    
                    model_id = model['id']
                    is_cloud = (
                        model.get('provider') in ['openai', 'anthropic'] or
                        model_id.startswith('gpt-') or
                        model_id.startswith('claude-')
                    )
                    
                    models[model_id] = {
                        **models.get(model_id, {}),
                        **model,
                        'loading': False,
                        'id': model_id,
                        'name': model.get('display_name') or model.get('name') or model_id,
                        'downloaded': model.get('ready') is True or model.get('downloaded') is True,
                        'provider': model.get('provider') or (
                            'openai' if model_id.startswith('gpt-') else
                            'anthropic' if model_id.startswith('claude-') else 'exo'
                        ),
                        'isLocalModel': not is_cloud,
                        'isCloudModel': is_cloud,
                        'canDelete': not is_cloud,
                        'download_percentage': model.get('download_percentage') or (100 if model.get('ready') else 0)
                    }
                    
            return models
        except Exception as e:
            print(f"Error fetching models: {e}")
            return {}
    
    with gr.Accordion("Models", open=True) as model_accordion:
        with gr.Row():
            show_downloaded_checkbox = gr.Checkbox(
                label="Show Downloaded Only", 
                value=False,
                container=True
            )
            refresh_btn = gr.Button("🔄", scale=1)
        
        # Container for model groups
        model_groups_container = gr.HTML(
            """<div class="model-loading">Loading models...</div>""",
            elem_classes=["model-groups-container"]
        )
    
    def update_model_display(models, downloaded_only=False, selected=""):
        """Update the HTML display of the models."""
        if not models:
            return """<div class="model-loading">No models available</div>"""
        
        grouped_models = group_models_by_prefix(models, downloaded_only)
        html_parts = ['<div class="model-groups">']
        
        # For each main group (LOCAL, CLOUD, NETWORK)
        for main_group, categories in grouped_models.items():
            html_parts.append(f'<div class="model-main-group">')
            html_parts.append(f'<div class="model-main-group-header">{main_group}</div>')
            html_parts.append(f'<div class="model-categories">')
            
            # For each category within the main group
            for category, category_models in categories.items():
                if not category_models:
                    continue
                
                html_parts.append(f'<div class="model-category">')
                html_parts.append(f'<div class="model-category-header">{category}</div>')
                html_parts.append(f'<div class="model-list">')
                
                # For each model in the category
                for model_id, model in category_models.items():
                    is_selected = model_id == selected
                    is_cloud = (model.get('isCloudModel') or 
                               model.get('provider') in ['openai', 'anthropic'])
                    is_peer = model.get('isPeerModel')
                    is_downloaded = (model.get('downloaded') is True or 
                                   model.get('ready') is True or 
                                   model.get('download_percentage') == 100 or
                                   is_cloud)
                    
                    download_percentage = model.get('download_percentage') or 0
                    total_size = model.get('total_size') or 0
                    
                    model_class = [
                        "model-item",
                        "selected" if is_selected else "",
                        "cloud-model" if is_cloud else "",
                        "peer-model" if is_peer else "",
                        "downloaded" if is_downloaded else ""
                    ]
                    
                    html_parts.append(f'<div class="{" ".join(model_class)}" data-model-id="{model_id}">')
                    
                    # Model header (name and badges)
                    html_parts.append(f'<div class="model-item-header">')
                    html_parts.append(f'<div class="model-name">{model.get("name", model_id)}')
                    
                    if is_cloud:
                        html_parts.append(f'<span class="model-badge cloud-badge">CLOUD</span>')
                    if is_peer:
                        html_parts.append(f'<span class="model-badge peer-badge">NETWORK</span>')
                    
                    html_parts.append(f'</div>')  # Close model name
                    
                    # Delete button for local models
                    if not is_cloud and not is_peer and model.get('canDelete') is not False and is_downloaded:
                        html_parts.append(f'<button class="model-delete-btn" data-model-id="{model_id}">🗑️</button>')
                    
                    html_parts.append(f'</div>')  # Close model header
                    
                    # Model info
                    html_parts.append(f'<div class="model-info">')
                    
                    # Download status
                    html_parts.append(f'<div class="model-status">')
                    if model.get('loading'):
                        html_parts.append(f'<span>Checking download status...</span>')
                    else:
                        status_text = "Downloaded" if is_downloaded else (
                            f"{download_percentage}% downloaded" if download_percentage > 0 else 
                            "Not downloaded"
                        )
                        html_parts.append(f'<span>{status_text}</span>')
                        
                        # Download button for non-downloaded local models
                        if (not is_downloaded and not model.get('loading') and 
                            download_percentage < 100 and not is_cloud and not is_peer):
                            btn_text = "Continue Download" if download_percentage > 0 else "Download"
                            html_parts.append(
                                f'<button class="model-download-btn" data-model-id="{model_id}">'
                                f'⬇️ {btn_text}</button>'
                            )
                    
                    html_parts.append(f'</div>')  # Close model status
                    
                    # Model size
                    if total_size:
                        html_parts.append(
                            f'<div class="model-size">{format_bytes(total_size)}</div>'
                        )
                    
                    html_parts.append(f'</div>')  # Close model info
                    html_parts.append(f'</div>')  # Close model item
                
                html_parts.append(f'</div>')  # Close model list
                html_parts.append(f'</div>')  # Close model category
            
            html_parts.append(f'</div>')  # Close model categories
            html_parts.append(f'</div>')  # Close main group
        
        html_parts.append('</div>')  # Close model groups
        
        # Add JavaScript for interaction
        html_parts.append('''
        <script>
            // Set up click handlers
            document.addEventListener('DOMContentLoaded', function() {
                // Model selection
                document.querySelectorAll('.model-item').forEach(item => {
                    item.addEventListener('click', function(e) {
                        if (!e.target.classList.contains('model-download-btn') && 
                            !e.target.classList.contains('model-delete-btn')) {
                            const modelId = this.getAttribute('data-model-id');
                            // Use gradio's function to update the state
                            if (window.gradioSelectedModel) {
                                window.gradioSelectedModel(modelId);
                            }
                        }
                    });
                });
                
                // Download buttons
                document.querySelectorAll('.model-download-btn').forEach(btn => {
                    btn.addEventListener('click', function(e) {
                        e.stopPropagation();
                        const modelId = this.getAttribute('data-model-id');
                        if (window.gradioDownloadModel) {
                            window.gradioDownloadModel(modelId);
                        }
                    });
                });
                
                // Delete buttons
                document.querySelectorAll('.model-delete-btn').forEach(btn => {
                    btn.addEventListener('click', function(e) {
                        e.stopPropagation();
                        const modelId = this.getAttribute('data-model-id');
                        if (window.gradioDeleteModel) {
                            window.gradioDeleteModel(modelId);
                        }
                    });
                });
            });
        </script>
        ''')
        
        return "".join(html_parts)
    
    def download_model(model_id):
        """Start downloading a model."""
        if on_download_model:
            on_download_model(model_id)
        return f"Started download of {model_id}"
    
    def delete_model(model_id):
        """Delete a downloaded model."""
        if on_delete_model:
            on_delete_model(model_id)
        return f"Deleted model {model_id}"
    
    def select_model(model_id):
        """Select a model and call the callback."""
        if on_model_select:
            on_model_select(model_id)
        return model_id
    
    # Register JavaScript functions
    js = """
    <script>
        window.gradioSelectedModel = function(modelId) {
            document.querySelectorAll('[id^="component-"]').forEach(el => {
                if (el.id.includes('select_model')) {
                    el.click();
                    setTimeout(() => {
                        const input = el.querySelector('input');
                        if (input) {
                            input.value = modelId;
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                        }
                    }, 100);
                }
            });
        };
        
        window.gradioDownloadModel = function(modelId) {
            document.querySelectorAll('[id^="component-"]').forEach(el => {
                if (el.id.includes('download_model')) {
                    el.click();
                    setTimeout(() => {
                        const input = el.querySelector('input');
                        if (input) {
                            input.value = modelId;
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                        }
                    }, 100);
                }
            });
        };
        
        window.gradioDeleteModel = function(modelId) {
            document.querySelectorAll('[id^="component-"]').forEach(el => {
                if (el.id.includes('delete_model')) {
                    el.click();
                    setTimeout(() => {
                        const input = el.querySelector('input');
                        if (input) {
                            input.value = modelId;
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                        }
                    }, 100);
                }
            });
        };
    </script>
    """
    
    gr.HTML(js, visible=False)
    
    # Setup event handlers
    def refresh_models():
        models = fetch_models()
        model_state.value = models
        return update_model_display(models, show_downloaded_only.value, selected_model.value)
    
    def toggle_downloaded_only(value):
        show_downloaded_only.value = value
        return update_model_display(model_state.value, value, selected_model.value)
    
    # Attach the download and delete functions to the HTML
    download_model_textbox = gr.Textbox(visible=False, elem_id="download_model")
    delete_model_textbox = gr.Textbox(visible=False, elem_id="delete_model")
    select_model_textbox = gr.Textbox(visible=False, elem_id="select_model")
    
    download_model_textbox.submit(download_model, inputs=[download_model_textbox], outputs=[gr.Textbox(visible=False)])
    delete_model_textbox.submit(delete_model, inputs=[delete_model_textbox], outputs=[gr.Textbox(visible=False)])
    select_model_textbox.submit(
        lambda x: (x, select_model(x)), 
        inputs=[select_model_textbox], 
        outputs=[selected_model, gr.Textbox(visible=False)]
    )
    
    # Set up event handlers
    refresh_btn.click(refresh_models, outputs=[model_groups_container])
    show_downloaded_checkbox.change(toggle_downloaded_only, inputs=[show_downloaded_checkbox], outputs=[model_groups_container])
    
    # Initial refresh
    refresh_models()
    
    return model_accordion, refresh_models, select_model