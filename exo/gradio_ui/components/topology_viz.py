"""
Network topology visualization component.
"""

from typing import Dict, Any, List, Optional
import json
import time

import gradio as gr
import plotly.graph_objects as go
import networkx as nx

from ..api import get_api_client


class TopologyVisualization:
    """
    Network topology visualization component for exo.
    """
    
    def __init__(self):
        """Initialize the topology visualization."""
        self.api_client = get_api_client()
        self.last_fetch_time = 0
        self.topology_data = None
        
    def fetch_topology(self) -> Dict[str, Any]:
        """
        Fetch topology data from the API.
        
        Returns:
            Topology data
        """
        current_time = time.time()
        # Only fetch every 5 seconds max
        if current_time - self.last_fetch_time < 5:
            return self.topology_data or {}
        
        try:
            self.topology_data = self.api_client.get_topology()
            self.last_fetch_time = current_time
            return self.topology_data
        except Exception as e:
            print(f"Error fetching topology: {e}")
            return {}
    
    def create_graph(self, topology_data: Dict[str, Any]) -> nx.Graph:
        """
        Create a NetworkX graph from topology data.
        
        Args:
            topology_data: Topology data from the API
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        if not topology_data or 'nodes' not in topology_data:
            return G
        
        # Add nodes
        for node in topology_data['nodes']:
            node_id = node.get('id')
            if not node_id:
                continue
                
            G.add_node(
                node_id,
                name=node.get('name', node_id),
                ip=node.get('ip', 'Unknown'),
                device_type=node.get('device_type', 'Unknown'),
                is_active=node.get('is_active', False)
            )
        
        # Add edges
        if 'links' in topology_data:
            for link in topology_data['links']:
                source = link.get('source')
                target = link.get('target')
                if source and target:
                    G.add_edge(
                        source, 
                        target, 
                        weight=link.get('weight', 1),
                        latency=link.get('latency', 0)
                    )
        
        return G
    
    def create_plotly_figure(self, G: nx.Graph) -> go.Figure:
        """
        Create a Plotly figure from a NetworkX graph.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Plotly figure
        """
        # If graph is empty, return empty figure
        if not G.nodes:
            fig = go.Figure()
            fig.update_layout(
                title="No nodes in network topology",
                template="plotly_dark"
            )
            return fig
        
        # Create positions
        pos = nx.spring_layout(G, seed=42)
        
        # Create node traces
        active_nodes = [n for n, attr in G.nodes(data=True) if attr.get('is_active')]
        inactive_nodes = [n for n, attr in G.nodes(data=True) if not attr.get('is_active')]
        
        # Edge trace
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.x += (x0, x1, None)
            edge_trace.y += (y0, y1, None)
        
        # Active nodes
        active_node_trace = go.Scatter(
            x=[pos[node][0] for node in active_nodes],
            y=[pos[node][1] for node in active_nodes],
            text=[f"{G.nodes[node].get('name', node)}<br>IP: {G.nodes[node].get('ip', 'Unknown')}<br>Device: {G.nodes[node].get('device_type', 'Unknown')}" 
                 for node in active_nodes],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color='green',
                size=15,
                line=dict(width=2, color='white')
            ),
            name='Active Nodes'
        )
        
        # Inactive nodes
        inactive_node_trace = go.Scatter(
            x=[pos[node][0] for node in inactive_nodes],
            y=[pos[node][1] for node in inactive_nodes],
            text=[f"{G.nodes[node].get('name', node)}<br>IP: {G.nodes[node].get('ip', 'Unknown')}<br>Device: {G.nodes[node].get('device_type', 'Unknown')}" 
                 for node in inactive_nodes],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color='red',
                size=15,
                line=dict(width=2, color='white')
            ),
            name='Inactive Nodes'
        )
        
        # Create figure
        data = [edge_trace]
        if active_nodes:
            data.append(active_node_trace)
        if inactive_nodes:
            data.append(inactive_node_trace)
            
        fig = go.Figure(data=data)
        
        # Configure layout
        fig.update_layout(
            title="Network Topology",
            titlefont_size=16,
            template="plotly_dark",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='closest',
            margin=dict(t=50, b=20, l=5, r=5),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def update_visualization(self) -> go.Figure:
        """
        Update the topology visualization.
        
        Returns:
            Plotly figure
        """
        topology_data = self.fetch_topology()
        G = self.create_graph(topology_data)
        return self.create_plotly_figure(G)


def create_topology_component() -> tuple[gr.Plot, TopologyVisualization]:
    """
    Create a topology visualization component.
    
    Returns:
        Tuple of (plot component, topology visualization instance)
    """
    topology = TopologyVisualization()
    
    with gr.Accordion("Network Topology", open=True) as topology_accordion:
        topology_plot = gr.Plot(
            topology.update_visualization,
            every=10  # Update every 10 seconds
        )
        refresh_btn = gr.Button("🔄 Refresh")
    
    refresh_btn.click(
        topology.update_visualization,
        outputs=topology_plot
    )
    
    return topology_plot, topology