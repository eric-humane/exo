document.addEventListener("alpine:init", () => {
  Alpine.data("state", () => ({
    // current state
    cstate: {
      time: null,
      messages: [],
      selectedModel: 'llama-3.2-1b',
    },

    // historical state
    histories: JSON.parse(localStorage.getItem("histories")) || [],

    home: 0,
    generating: false,
    endpoint: `${window.location.origin}/v1`,

    // Initialize error message structure
    errorMessage: null,
    errorExpanded: false,
    errorTimeout: null,

    // performance tracking
    time_till_first: 0,
    tokens_per_second: 0,
    total_tokens: 0,

    // image handling
    imagePreview: null,

    // download progress
    downloadProgress: null,

    // Pending message storage
    pendingMessage: null,

    // Add models state alongside existing state
    models: {},

    // Show only models available locally
    showDownloadedOnly: localStorage.getItem("showDownloadedOnly") === "true" || false,

    topology: null,

    // Add these new properties
    expandedGroups: {
      'LOCAL': true,
      'CLOUD': true,
      'NETWORK': true
    },
    
    // Timestamp of last fetches to prevent excessive polling
    lastModelFetch: 0,
    lastProgressFetch: 0,

    init() {
      // Clean up any pending messages
      localStorage.removeItem("pendingMessage");
      
      // Initial model fetch - once at startup
      this.fetchModels();
      
      // Setup manual refresh button functionality when the button exists
      document.getElementById('refresh-models')?.addEventListener('click', () => {
        this.fetchModels();
      });
      
      // Check download progress once at startup
      this.fetchDownloadProgress();
      
      // Setup network topology check periodically
      this.fetchTopology();
      setInterval(() => this.fetchTopology(), 10000);
      
      // Only fetch new data when the user is actively interacting with the page
      window.addEventListener('focus', () => this.onPageFocus());
      
      // Add event listener for visibility changes
      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
          this.onPageFocus();
        }
      });
    },
    
    filterChanged() {
      // Save filter preference to localStorage
      localStorage.setItem("showDownloadedOnly", this.showDownloadedOnly);
      this.fetchModels();
    },
    
    async fetchTopology() {
      try {
        const requestOptions = {
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
          }
        };
        
        const response = await fetch(`${window.location.origin}/topology`, requestOptions);
        if (response.ok) {
          const data = await response.json();
          this.topology = data;
          
          // Render topology visualization
          this.renderTopology();
        }
      } catch (error) {
        console.error('Error fetching topology:', error);
      }
    },
    
    renderTopology() {
      if (!this.topology || !this.$refs.topologyViz) return;
      
      // Clear previous content
      this.$refs.topologyViz.innerHTML = '';
      
      // If there are no nodes, show empty message
      if (!this.topology.nodes || this.topology.nodes.length === 0) {
        const emptyMessage = document.createElement('div');
        emptyMessage.className = 'empty-category-message';
        emptyMessage.textContent = 'No nodes in network topology';
        this.$refs.topologyViz.appendChild(emptyMessage);
        return;
      }
      
      // Create node elements
      this.topology.nodes.forEach(node => {
        const nodeElement = document.createElement('div');
        nodeElement.className = 'topology-node';
        
        // Node header
        const nodeHeader = document.createElement('div');
        nodeHeader.className = 'node-info';
        
        const statusIndicator = document.createElement('div');
        statusIndicator.className = 'status ' + (node.is_active ? 'active' : 'inactive');
        
        const nodeName = document.createElement('span');
        nodeName.textContent = node.name || node.id || 'Unknown Node';
        
        nodeHeader.appendChild(statusIndicator);
        nodeHeader.appendChild(nodeName);
        nodeElement.appendChild(nodeHeader);
        
        // Node details
        if (node.ip || node.device_type) {
          const nodeDetails = document.createElement('div');
          nodeDetails.className = 'node-details';
          
          if (node.ip) {
            const ipElement = document.createElement('span');
            ipElement.innerHTML = `<i class="fas fa-network-wired"></i> ${node.ip}`;
            nodeDetails.appendChild(ipElement);
          }
          
          if (node.device_type) {
            const deviceElement = document.createElement('span');
            deviceElement.innerHTML = `<i class="fas fa-microchip"></i> ${node.device_type}`;
            nodeDetails.appendChild(deviceElement);
          }
          
          nodeElement.appendChild(nodeDetails);
        }
        
        // Add to visualization
        this.$refs.topologyViz.appendChild(nodeElement);
      });
    },
    
    onPageFocus() {
      // Only fetch if it's been at least 5 seconds since the last fetch
      const now = Date.now();
      if (now - this.lastModelFetch >= 5000) {
        this.fetchModels();
      }
      
      if (now - this.lastProgressFetch >= 2000) {
        this.fetchDownloadProgress();
      }
    },

    async fetchModels() {
      try {
        this.lastModelFetch = Date.now();
        
        // Clear loading state for all models
        Object.keys(this.models).forEach(key => {
          if (this.models[key]) {
            this.models[key].loading = false;
          }
        });
        
        // Step 1: Add default cloud models to ensure they always show up
        this.ensureDefaultCloudModels();
        
        // Configure standard request options for CORS compatibility
        const requestOptions = {
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
          }
        };
        
        // Step 2: Get local models from the initial_models endpoint
        try {
          const initialResponse = await fetch(`${window.location.origin}/initial_models`, requestOptions);
          if (initialResponse.ok) {
            const initialModels = await initialResponse.json();
            Object.entries(initialModels).forEach(([modelId, modelData]) => {
              // If the model doesn't exist yet, create it
              if (!this.models[modelId]) {
                this.models[modelId] = {};
              }
              
              // Update with local model data
              this.models[modelId] = {
                ...this.models[modelId],
                ...modelData,
                id: modelId,
                name: modelData.name || modelData.display_name || modelId,
                loading: false,
                provider: 'exo',
                isLocalModel: true,
                total_size: modelData.total_size || 0,
                download_percentage: modelData.download_percentage || 0,
                downloaded: modelData.downloaded === true,
                canDelete: true
              };
            });
          }
        } catch (error) {
          console.error('Error fetching initial models:', error);
        }
        
        // Step 3: Get the complete model list including cloud models
        try {
          const response = await fetch(`${window.location.origin}/models`, requestOptions);
          
          if (response.ok) {
            const data = await response.json();
            if (data && data.data && Array.isArray(data.data)) {
              // Process model data from response
              data.data.forEach(model => {
                // Make sure the model has an ID
                if (!model.id) return;
                
                // Determine if it's a cloud model
                const isCloud = model.provider === 'openai' || 
                               model.provider === 'anthropic' ||
                               model.id.startsWith('gpt-') || 
                               model.id.startsWith('claude-');
                
                // If the model doesn't exist yet, create it
                if (!this.models[model.id]) {
                  this.models[model.id] = {};
                }
                
                // Update the model data
                this.models[model.id] = {
                  ...this.models[model.id],
                  ...model,
                  loading: false,
                  id: model.id,
                  name: model.display_name || model.name || model.id,
                  downloaded: model.ready === true || model.downloaded === true,
                  provider: model.provider || (model.id.startsWith('gpt-') ? 'openai' :
                                             model.id.startsWith('claude-') ? 'anthropic' : 'exo'),
                  isLocalModel: !isCloud,
                  isCloudModel: isCloud,
                  canDelete: !isCloud,
                  download_percentage: model.download_percentage || (model.ready ? 100 : 0)
                };
              });
            }
          }
        } catch (error) {
          console.error('Error fetching models API:', error);
        }
        
        // Force Alpine.js to update
        this.models = {...this.models};
      } catch (error) {
        console.error('Error in fetchModels:', error);
      }
    },
    
    ensureDefaultCloudModels() {
      // OpenAI models
      const openaiModels = [
        {
          id: 'gpt-4',
          name: 'GPT-4',
          provider: 'openai',
          isCloudModel: true,
          canDelete: false,
          downloaded: true,
          download_percentage: 100
        },
        {
          id: 'gpt-4-turbo',
          name: 'GPT-4 Turbo',
          provider: 'openai',
          isCloudModel: true,
          canDelete: false,
          downloaded: true,
          download_percentage: 100
        },
        {
          id: 'gpt-3.5-turbo',
          name: 'GPT-3.5 Turbo',
          provider: 'openai',
          isCloudModel: true,
          canDelete: false,
          downloaded: true,
          download_percentage: 100
        }
      ];
      
      // Anthropic models
      const anthropicModels = [
        {
          id: 'claude-3-opus',
          name: 'Claude 3 Opus',
          provider: 'anthropic',
          isCloudModel: true,
          canDelete: false,
          downloaded: true,
          download_percentage: 100
        },
        {
          id: 'claude-3-sonnet',
          name: 'Claude 3 Sonnet',
          provider: 'anthropic',
          isCloudModel: true,
          canDelete: false,
          downloaded: true,
          download_percentage: 100
        },
        {
          id: 'claude-3-haiku',
          name: 'Claude 3 Haiku',
          provider: 'anthropic',
          isCloudModel: true,
          canDelete: false,
          downloaded: true,
          download_percentage: 100
        }
      ];
      
      // Add all OpenAI models
      openaiModels.forEach(model => {
        if (!this.models[model.id]) {
          this.models[model.id] = model;
        } else {
          this.models[model.id] = {...this.models[model.id], ...model};
        }
      });
      
      // Add all Anthropic models
      anthropicModels.forEach(model => {
        if (!this.models[model.id]) {
          this.models[model.id] = model;
        } else {
          this.models[model.id] = {...this.models[model.id], ...model};
        }
      });
    },

    removeHistory(cstate) {
      const index = this.histories.findIndex((state) => {
        return state.time === cstate.time;
      });
      if (index !== -1) {
        this.histories.splice(index, 1);
        localStorage.setItem("histories", JSON.stringify(this.histories));
      }
    },

    clearAllHistory() {
      this.histories = [];
      localStorage.setItem("histories", JSON.stringify([]));
    },

    // Utility functions
    formatBytes(bytes) {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    formatDuration(seconds) {
      if (seconds === null || seconds === undefined || isNaN(seconds)) return '';
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = Math.floor(seconds % 60);
      if (h > 0) return `${h}h ${m}m ${s}s`;
      if (m > 0) return `${m}m ${s}s`;
      return `${s}s`;
    },

    async handleImageUpload(event) {
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          this.imagePreview = e.target.result;
          this.imageUrl = e.target.result; // Store the image URL
          // Add image preview to the chat
          this.cstate.messages.push({
            role: "user",
            content: `![Uploaded Image](${this.imagePreview})`,
          });
        };
        reader.readAsDataURL(file);
      }
    },

    async handleSend() {
      try {
        const el = document.getElementById("input-form");
        const value = el.value.trim();
        if (!value && !this.imagePreview) return;

        if (this.generating) return;
        this.generating = true;
        if (this.home === 0) this.home = 1;

        // ensure that going back in history will go back to home
        window.history.pushState({}, "", "/");

        // add message to list
        if (value) {
          this.cstate.messages.push({ role: "user", content: value });
        }

        // clear textarea
        el.value = "";
        el.style.height = "auto";
        el.style.height = el.scrollHeight + "px";

        localStorage.setItem("pendingMessage", value);
        this.processMessage(value);
      } catch (error) {
        console.error('error', error);
        this.setError(error);
        this.generating = false;
      }
    },

    async processMessage(value) {
      try {
        // reset performance tracking
        const prefill_start = Date.now();
        let start_time = 0;
        let tokens = 0;
        this.tokens_per_second = 0;

        // prepare messages for API request
        let apiMessages = this.cstate.messages.map(msg => {
          if (msg.content.startsWith('![Uploaded Image]')) {
            return {
              role: "user",
              content: [
                {
                  type: "image_url",
                  image_url: {
                    url: this.imageUrl
                  }
                },
                {
                  type: "text",
                  text: value // Use the actual text the user typed
                }
              ]
            };
          } else {
            return {
              role: msg.role,
              content: msg.content
            };
          }
        });
        
        if (this.cstate.selectedModel === "stable-diffusion-2-1-base") {
          // Send a request to the image generation endpoint
          console.log(apiMessages[apiMessages.length - 1].content)
          console.log(this.cstate.selectedModel)  
          console.log(this.endpoint)
          const response = await fetch(`${this.endpoint}/image/generations`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              "model": 'stable-diffusion-2-1-base',
              "prompt": apiMessages[apiMessages.length - 1].content,
              "image_url": this.imageUrl
            }),
          });
      
          if (!response.ok) {
            throw new Error("Failed to fetch");
          }
          const reader = response.body.getReader();
          let done = false;
          let gottenFirstChunk = false;
  
          while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            const decoder = new TextDecoder();
  
            if (value) {
              // Assume non-binary data (text) comes first
              const chunk = decoder.decode(value, { stream: true });
              const parsed = JSON.parse(chunk);
              console.log(parsed)
  
              if (parsed.progress) {
                if (!gottenFirstChunk) {
                  this.cstate.messages.push({ role: "assistant", content: "" });
                  gottenFirstChunk = true;
                }
                this.cstate.messages[this.cstate.messages.length - 1].content = parsed.progress;
              }
              else if (parsed.images) {
                if (!gottenFirstChunk) {
                  this.cstate.messages.push({ role: "assistant", content: "" });
                  gottenFirstChunk = true;
                }
                const imageUrl = parsed.images[0].url;
                console.log(imageUrl)
                this.cstate.messages[this.cstate.messages.length - 1].content = `![Generated Image](${imageUrl}?t=${Date.now()})`;
              }
            }
          }
        }
        
        else{        
          const containsImage = apiMessages.some(msg => Array.isArray(msg.content) && msg.content.some(item => item.type === 'image_url'));
          if (containsImage) {
            // Map all messages with string content to object with type text
            apiMessages = apiMessages.map(msg => {
              if (typeof msg.content === 'string') {
                return {
                  ...msg,
                  content: [
                    {
                      type: "text",
                      text: msg.content
                    }
                  ]
                };
              }
              return msg;
            });
          }

          console.log(apiMessages)
          //start receiving server sent events
          let gottenFirstChunk = false;
          for await (
            const chunk of this.openaiChatCompletion(this.cstate.selectedModel, apiMessages)
          ) {
            if (!gottenFirstChunk) {
              this.cstate.messages.push({ role: "assistant", content: "" });
              gottenFirstChunk = true;
            }

            // add chunk to the last message
            this.cstate.messages[this.cstate.messages.length - 1].content += chunk;

            // calculate performance tracking
            tokens += 1;
            this.total_tokens += 1;
            if (start_time === 0) {
              start_time = Date.now();
              this.time_till_first = start_time - prefill_start;
            } else {
              const diff = Date.now() - start_time;
              if (diff > 0) {
                this.tokens_per_second = tokens / (diff / 1000);
              }
            }
          }
        }
        // Clean the cstate before adding it to histories
        const cleanedCstate = JSON.parse(JSON.stringify(this.cstate));
        cleanedCstate.messages = cleanedCstate.messages.map(msg => {
          if (Array.isArray(msg.content)) {
            return {
              ...msg,
              content: msg.content.map(item =>
                item.type === 'image_url' ? { type: 'image_url', image_url: { url: '[IMAGE_PLACEHOLDER]' } } : item
              )
            };
          }
          return msg;
        });

        // Update the state in histories or add it if it doesn't exist
        const index = this.histories.findIndex((cstate) => cstate.time === cleanedCstate.time);
        cleanedCstate.time = Date.now();
        if (index !== -1) {
          // Update the existing entry
          this.histories[index] = cleanedCstate;
        } else {
          // Add a new entry
          this.histories.push(cleanedCstate);
        }
        console.log(this.histories)
        // update in local storage
        try {
          localStorage.setItem("histories", JSON.stringify(this.histories));
        } catch (error) {
          console.error("Failed to save histories to localStorage:", error);
        }
      } catch (error) {
        console.error('error', error);
        this.setError(error);
      } finally {
        this.generating = false;
      }
    },

    async handleEnter(event) {
      // if shift is not pressed
      if (!event.shiftKey) {
        event.preventDefault();
        await this.handleSend();
      }
    },

    updateTotalTokens(messages) {
      fetch(`${this.endpoint}/chat/token/encode`, {
        method: "POST",
        credentials: 'include',
        headers: { 
          "Content-Type": "application/json",
          "X-Requested-With": "XMLHttpRequest"
        },
        body: JSON.stringify({ messages }),
      }).then((response) => response.json()).then((data) => {
        this.total_tokens = data.length;
      }).catch(console.error);
    },

    async *openaiChatCompletion(model, messages) {
      const response = await fetch(`${this.endpoint}/chat/completions`, {
        method: "POST",
        credentials: 'include',
        headers: {
          "Content-Type": "application/json",
          "X-Requested-With": "XMLHttpRequest"
        },
        body: JSON.stringify({
          "model": model,
          "messages": messages,
          "stream": true,
        }),
      });
      if (!response.ok) {
        const errorResBody = await response.json()
        if (errorResBody?.detail) {
          throw new Error(`Failed to fetch completions: ${errorResBody.detail}`);
        } else {
          throw new Error("Failed to fetch completions: Unknown error");
        }
      }

      const reader = response.body.pipeThrough(new TextDecoderStream())
        .pipeThrough(new EventSourceParserStream()).getReader();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        if (value.type === "event") {
          try {
            const json = JSON.parse(value.data);
            if (json.done === true) {
              // Handle the done message with proper JSON format
              break;
            }
            if (json.choices) {
              const choice = json.choices[0];
              if (choice.finish_reason === "stop") break;
              if (choice.delta.content) yield choice.delta.content;
            }
          } catch (e) {
            // Handle case where [DONE] is sent as plain string (legacy format)
            if (value.data === "[DONE]") break;
            console.error("Error parsing event data:", e, "Data:", value.data);
          }
        }
      }
    },

    async fetchDownloadProgress() {
      try {
        this.lastProgressFetch = Date.now();
        
        const requestOptions = {
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
          }
        };
        
        const response = await fetch(`${this.endpoint}/download/progress`, requestOptions);
        
        if (!response.ok) return;
        
        const data = await response.json();
        if (!data || typeof data !== 'object') return;
        
        const progressArray = Object.values(data);
        if (progressArray.length === 0) {
          this.downloadProgress = null;
          return;
        }
        
        this.downloadProgress = progressArray.map(progress => {
          if (!progress || typeof progress !== 'object') {
            return { isComplete: false, percentage: 0 };
          }
          
          // Check if download is complete
          if (progress.status === "complete") {
            return {
              ...progress,
              isComplete: true,
              percentage: 100
            };
          } else if (progress.status === "failed") {
            return {
              ...progress,
              isComplete: false,
              errorMessage: "Download failed",
              percentage: 0
            };
          } else {
            const downloadedBytes = typeof progress.downloaded_bytes === 'number' ? progress.downloaded_bytes : 0;
            const totalBytes = typeof progress.total_bytes === 'number' && progress.total_bytes > 0 ? progress.total_bytes : 1;
            const percentage = Math.min(((downloadedBytes / totalBytes) * 100), 99.9).toFixed(2);
            
            return {
              ...progress,
              isComplete: false,
              downloaded_bytes_display: this.formatBytes(downloadedBytes),
              total_bytes_display: this.formatBytes(totalBytes),
              overall_speed_display: progress.overall_speed ? this.formatBytes(progress.overall_speed) + '/s' : '',
              overall_eta_display: progress.overall_eta ? this.formatDuration(progress.overall_eta) : '',
              percentage: percentage
            };
          }
        });
        
        // If all downloads are complete, update models and clear progress
        const allComplete = this.downloadProgress.every(progress => progress.isComplete);
        if (allComplete) {
          this.fetchModels(); // Refresh models after download completes
          this.downloadProgress = null;
        }
        
        // If downloads are in progress, check again soon
        if (this.downloadProgress !== null) {
          setTimeout(() => this.fetchDownloadProgress(), 2000);
        }
      } catch (error) {
        console.error("Error fetching download progress:", error);
      }
    },

    initTopology() {
      // Initial fetch is handled in init()
      // This just ensures the reference to the element is available
      // Later operations will use this.$refs.topologyViz
      console.log("Topology reference initialized");
    },
    
    // Add a helper method to set errors consistently
    setError(error) {
      this.errorMessage = {
        basic: error.message || "An unknown error occurred",
        stack: error.stack || ""
      };
      this.errorExpanded = false;

      if (this.errorTimeout) {
        clearTimeout(this.errorTimeout);
      }

      if (!this.errorExpanded) {
        this.errorTimeout = setTimeout(() => {
          this.errorMessage = null;
          this.errorExpanded = false;
        }, 30 * 1000);
      }
    },

    async deleteModel(modelName, model) {
      // Check if this is a cloud model - if so, don't allow deletion
      if (model.isCloudModel || model.isPeerModel || model.canDelete === false) {
        this.setError({
          message: `${model.name} cannot be deleted because it's not stored locally.`
        });
        return;
      }
      
      const downloadedSize = model.total_downloaded || 0;
      const sizeMessage = downloadedSize > 0 ?
        `This will free up ${this.formatBytes(downloadedSize)} of space.` :
        'This will remove any partially downloaded files.';

      if (!confirm(`Are you sure you want to delete ${model.name}? ${sizeMessage}`)) {
        return;
      }

      try {
        const response = await fetch(`${window.location.origin}/models/${modelName}`, {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json'
          }
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Failed to delete model');
        }

        // Update the model status in the UI
        if (this.models[modelName]) {
          this.models[modelName].downloaded = false;
          this.models[modelName].download_percentage = 0;
          this.models[modelName].total_downloaded = 0;
        }

        // If this was the selected model, switch to a different one
        if (this.cstate.selectedModel === modelName) {
          const availableModel = Object.keys(this.models).find(key => this.models[key].downloaded);
          this.cstate.selectedModel = availableModel || 'llama-3.2-1b';
        }

        // Show success message
        console.log(`Model deleted successfully from: ${data.path}`);

        // Refresh the model list
        this.fetchModels();
      } catch (error) {
        console.error('Error deleting model:', error);
        this.setError(error.message || 'Failed to delete model');
      }
    },

    async handleDownload(modelName) {
      try {
        const response = await fetch(`${window.location.origin}/download`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: modelName
          })
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || 'Failed to start download');
        }

        // Update the model's status immediately when download starts
        if (this.models[modelName]) {
          this.models[modelName] = {
            ...this.models[modelName],
            loading: true
          };
        }

        // Start checking download progress
        this.fetchDownloadProgress();

      } catch (error) {
        console.error('Error starting download:', error);
        this.setError(error);
      }
    },

    // Update the groupModelsByPrefix method to include provider info and separate online models
    groupModelsByPrefix(models) {
      // Create models structure
      const groups = {
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
        'NETWORK': {} // Network/peer models
      };
      
      console.log('Processing models, total count:', Object.keys(models).length);
      
      // First, process all models to ensure structure is correct
      Object.entries(models).forEach(([modelId, model]) => {
        // Skip models with no data
        if (!model || !modelId) return;
        
        try {
          // Cloud models
          if (model.isCloudModel || model.provider === 'openai' || model.provider === 'anthropic' || 
              modelId.startsWith('gpt-') || modelId.startsWith('claude-')) {
            
            const provider = model.provider ? model.provider.toUpperCase() : 
                            modelId.startsWith('gpt-') ? 'OPENAI' : 
                            modelId.startsWith('claude-') ? 'ANTHROPIC' : 'OTHER';
            
            if (!groups['CLOUD'][provider]) {
              groups['CLOUD'][provider] = {};
            }
            
            groups['CLOUD'][provider][modelId] = model;
          }
          // Network models
          else if (model.isPeerModel || model.peer_id || model.node_id) {
            const peerName = model.peer_name || model.node_name || 'UNKNOWN';
            
            if (!groups['NETWORK'][peerName]) {
              groups['NETWORK'][peerName] = {};
            }
            
            groups['NETWORK'][peerName][modelId] = model;
          }
          // Local models
          else {
            // Parse model category from name
            let category = 'OTHER';
            if (modelId.includes('-')) {
              const parts = modelId.split('-');
              category = parts[0].toUpperCase();
            }
            
            // Ensure the category exists
            if (!groups['LOCAL'][category]) {
              groups['LOCAL'][category] = {};
            }
            
            // Add the model to its category
            groups['LOCAL'][category][modelId] = model;
          }
        } catch (error) {
          console.error('Error processing model:', modelId, error);
        }
      });
      
      // Apply downloaded-only filter if enabled
      if (this.showDownloadedOnly) {
        console.log('Applying downloaded-only filter');
        
        // Process each group type
        Object.keys(groups).forEach(groupType => {
          // Process each category in the group
          Object.keys(groups[groupType]).forEach(category => {
            // Filter models in this category
            const categoryModels = groups[groupType][category];
            const filteredCategoryModels = {};
            
            Object.entries(categoryModels).forEach(([modelId, model]) => {
              // For cloud models, they're always "downloaded"
              if (model.provider && model.provider !== 'exo') {
                filteredCategoryModels[modelId] = model;
              }
              // For local models, check downloaded status (include ready models)
              else if (model.downloaded === true || model.ready === true || model.download_percentage === 100) {
                filteredCategoryModels[modelId] = model;
              }
            });
            
            // Replace with filtered models
            groups[groupType][category] = filteredCategoryModels;
          });
        });
      }
      
      // Clean up empty categories, but keep necessary structure
      Object.keys(groups).forEach(groupType => {
        // Never delete the top level LOCAL or CLOUD categories
        Object.keys(groups[groupType]).forEach(category => {
          // Only delete empty subcategories that are not essential
          if (Object.keys(groups[groupType][category]).length === 0 && 
             !(groupType === 'LOCAL' && ['LLAMA', 'MISTRAL', 'DEEPSEEK'].includes(category)) &&
             !(groupType === 'CLOUD' && ['OPENAI', 'ANTHROPIC'].includes(category))) {
            delete groups[groupType][category];
          }
        });
      });
      
      return groups;
    },

    // Helper methods to count downloaded models
    countDownloadedModels(models) {
      if (!models || typeof models !== 'object') {
        return 0;
      }
      const count = Object.values(models)
        .filter(model => model && typeof model === 'object' && 
                (model.downloaded === true || 
                 model.download_percentage === 100 || 
                 model.ready === true || 
                 (model.provider && model.provider !== 'exo')))
        .length;
      return count;
    },

    getGroupCounts(groupModels) {
      if (!groupModels || typeof groupModels !== 'object') {
        return '[0/0]';
      }
      const total = Object.keys(groupModels).length;
      const downloaded = this.countDownloadedModels(groupModels);
      return downloaded > 0 ? `[${downloaded}/${total}]` : `[0/${total}]`;
    },
    
    // Safer version for nested groups
    getNestedGroupCounts(subGroups) {
      if (!subGroups || typeof subGroups !== 'object') {
        return '[0/0]';
      }
      
      try {
        const allModels = Object.values(subGroups)
          .filter(group => group && typeof group === 'object')
          .flatMap(group => Object.values(group || {}));
            
        const total = allModels.length;
        const downloaded = allModels.filter(model => 
            model && 
            (model.downloaded === true || 
             model.download_percentage === 100 || 
             model.ready === true || 
             (model.provider && model.provider !== 'exo'))
        ).length;
        return downloaded > 0 ? `[${downloaded}/${total}]` : `[0/${total}]`;
      } catch (error) {
        console.error("Error calculating group counts:", error);
        return '[?/?]';
      }
    },

    toggleGroup(prefix, subPrefix = null) {
      const key = subPrefix ? `${prefix}-${subPrefix}` : prefix;
      this.expandedGroups[key] = !this.expandedGroups[key];
    },

    isGroupExpanded(prefix, subPrefix = null) {
      const key = subPrefix ? `${prefix}-${subPrefix}` : prefix;
      return this.expandedGroups[key] || false;
    },
  }));
});

const { markedHighlight } = globalThis.markedHighlight;
marked.use(markedHighlight({
  langPrefix: "hljs language-",
  highlight(code, lang, _info) {
    const language = hljs.getLanguage(lang) ? lang : "plaintext";
    return hljs.highlight(code, { language }).value;
  },
}));

// **** eventsource-parser ****
class EventSourceParserStream extends TransformStream {
  constructor() {
    let parser;

    super({
      start(controller) {
        parser = createParser((event) => {
          if (event.type === "event") {
            controller.enqueue(event);
          }
        });
      },

      transform(chunk) {
        parser.feed(chunk);
      },
    });
  }
}

function createParser(onParse) {
  let isFirstChunk;
  let buffer;
  let startingPosition;
  let startingFieldLength;
  let eventId;
  let eventName;
  let data;
  reset();
  return {
    feed,
    reset,
  };
  function reset() {
    isFirstChunk = true;
    buffer = "";
    startingPosition = 0;
    startingFieldLength = -1;
    eventId = void 0;
    eventName = void 0;
    data = "";
  }
  function feed(chunk) {
    buffer = buffer ? buffer + chunk : chunk;
    if (isFirstChunk && hasBom(buffer)) {
      buffer = buffer.slice(BOM.length);
    }
    isFirstChunk = false;
    const length = buffer.length;
    let position = 0;
    let discardTrailingNewline = false;
    while (position < length) {
      if (discardTrailingNewline) {
        if (buffer[position] === "\n") {
          ++position;
        }
        discardTrailingNewline = false;
      }
      let lineLength = -1;
      let fieldLength = startingFieldLength;
      let character;
      for (
        let index = startingPosition;
        lineLength < 0 && index < length;
        ++index
      ) {
        character = buffer[index];
        if (character === ":" && fieldLength < 0) {
          fieldLength = index - position;
        } else if (character === "\r") {
          discardTrailingNewline = true;
          lineLength = index - position;
        } else if (character === "\n") {
          lineLength = index - position;
        }
      }
      if (lineLength < 0) {
        startingPosition = length - position;
        startingFieldLength = fieldLength;
        break;
      } else {
        startingPosition = 0;
        startingFieldLength = -1;
      }
      parseEventStreamLine(buffer, position, fieldLength, lineLength);
      position += lineLength + 1;
    }
    if (position === length) {
      buffer = "";
    } else if (position > 0) {
      buffer = buffer.slice(position);
    }
  }
  function parseEventStreamLine(lineBuffer, index, fieldLength, lineLength) {
    if (lineLength === 0) {
      if (data.length > 0) {
        onParse({
          type: "event",
          id: eventId,
          event: eventName || void 0,
          data: data.slice(0, -1),
          // remove trailing newline
        });

        data = "";
        eventId = void 0;
      }
      eventName = void 0;
      return;
    }
    const noValue = fieldLength < 0;
    const field = lineBuffer.slice(
      index,
      index + (noValue ? lineLength : fieldLength),
    );
    let step = 0;
    if (noValue) {
      step = lineLength;
    } else if (lineBuffer[index + fieldLength + 1] === " ") {
      step = fieldLength + 2;
    } else {
      step = fieldLength + 1;
    }
    const position = index + step;
    const valueLength = lineLength - step;
    const value = lineBuffer.slice(position, position + valueLength).toString();
    if (field === "data") {
      data += value ? "".concat(value, "\n") : "\n";
    } else if (field === "event") {
      eventName = value;
    } else if (field === "id" && !value.includes("\0")) {
      eventId = value;
    } else if (field === "retry") {
      const retry = parseInt(value, 10);
      if (!Number.isNaN(retry)) {
        onParse({
          type: "reconnect-interval",
          value: retry,
        });
      }
    }
  }
}

const BOM = [239, 187, 191];
function hasBom(buffer) {
  return BOM.every((charCode, index) => buffer.charCodeAt(index) === charCode);
}