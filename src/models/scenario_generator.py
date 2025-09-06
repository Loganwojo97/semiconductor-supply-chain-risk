"""
Synthetic Scenario Generator for Supply Chain Risk Analysis
This module creates realistic what-if scenarios and simulates risk propagation
through the semiconductor supply chain network using graph algorithms and ML.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class DisruptionType(Enum):
    """Types of supply chain disruptions"""
    NATURAL_DISASTER = "natural_disaster"
    GEOPOLITICAL = "geopolitical"
    CYBER_ATTACK = "cyber_attack"
    PANDEMIC = "pandemic"
    FINANCIAL_CRISIS = "financial_crisis"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    LABOR_STRIKE = "labor_strike"
    TRADE_RESTRICTION = "trade_restriction"

class PropagationModel(Enum):
    """Risk propagation models"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SIGMOID = "sigmoid"
    THRESHOLD = "threshold"
    STOCHASTIC = "stochastic"

@dataclass
class SupplyChainNode:
    """Represents a node in the supply chain network"""
    id: str
    name: str
    type: str  # manufacturer, supplier, logistics, customer
    location: Dict[str, float]  # lat, lon, country, region
    capacity: float = 100.0
    current_load: float = 70.0
    risk_score: float = 0.0
    resilience_factor: float = 0.5  # 0-1, ability to absorb shocks
    inventory_days: int = 30
    alternative_suppliers: List[str] = field(default_factory=list)
    critical_components: List[str] = field(default_factory=list)
    financial_health: float = 0.7  # 0-1 score
    
@dataclass
class SupplyChainEdge:
    """Represents a connection in the supply chain"""
    source: str
    target: str
    capacity: float
    current_flow: float
    lead_time_days: int
    dependency_strength: float  # 0-1, how much target depends on source
    transportation_mode: str  # sea, air, rail, road
    alternative_routes: int = 0
    
@dataclass
class Scenario:
    """Defines a what-if scenario"""
    id: str
    name: str
    description: str
    disruption_type: DisruptionType
    epicenter_nodes: List[str]
    initial_impact: float  # 0-100
    duration_days: int
    propagation_model: PropagationModel
    propagation_speed: float  # 0-1, how fast risk spreads
    recovery_rate: float  # 0-1, how fast nodes recover
    external_factors: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class SupplyChainNetworkSimulator:
    """
    Advanced supply chain network simulator with ML-enhanced risk propagation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.network = nx.DiGraph()
        self.nodes: Dict[str, SupplyChainNode] = {}
        self.edges: Dict[Tuple[str, str], SupplyChainEdge] = {}
        self.current_risks: Dict[str, float] = {}
        self.risk_history: List[Dict] = []
        self.scenario_results: Dict[str, Any] = {}
        
    def _default_config(self) -> Dict:
        return {
            'max_propagation_depth': 5,
            'risk_threshold': 0.7,
            'simulation_timesteps': 100,
            'monte_carlo_runs': 1000,
            'network_density': 0.3,
            'cascade_probability': 0.6
        }
    
    def build_semiconductor_network(self) -> nx.DiGraph:
        """Build a realistic semiconductor supply chain network"""
        
        # Define key semiconductor supply chain nodes
        nodes_data = [
            # Foundries
            SupplyChainNode("TSMC_TW", "TSMC Taiwan", "foundry", 
                          {"lat": 24.8, "lon": 120.9, "country": "Taiwan", "region": "Asia"},
                          capacity=100, resilience_factor=0.7, inventory_days=45),
            SupplyChainNode("SAMSUNG_KR", "Samsung Foundry", "foundry",
                          {"lat": 37.5, "lon": 127.0, "country": "South Korea", "region": "Asia"},
                          capacity=85, resilience_factor=0.8, inventory_days=40),
            SupplyChainNode("INTEL_US", "Intel Fabs", "foundry",
                          {"lat": 45.5, "lon": -122.8, "country": "USA", "region": "Americas"},
                          capacity=70, resilience_factor=0.75, inventory_days=35),
            
            # Equipment manufacturers
            SupplyChainNode("ASML_NL", "ASML", "equipment",
                          {"lat": 51.4, "lon": 5.4, "country": "Netherlands", "region": "Europe"},
                          capacity=100, resilience_factor=0.9, inventory_days=60,
                          critical_components=["EUV_lithography"]),
            SupplyChainNode("AMAT_US", "Applied Materials", "equipment",
                          {"lat": 37.3, "lon": -122.0, "country": "USA", "region": "Americas"},
                          capacity=90, resilience_factor=0.8, inventory_days=50),
            
            # Material suppliers
            SupplyChainNode("SHIN_ETSU_JP", "Shin-Etsu Chemical", "materials",
                          {"lat": 35.6, "lon": 139.7, "country": "Japan", "region": "Asia"},
                          capacity=95, resilience_factor=0.7, inventory_days=30,
                          critical_components=["silicon_wafers"]),
            SupplyChainNode("SUMCO_JP", "SUMCO", "materials",
                          {"lat": 35.7, "lon": 139.8, "country": "Japan", "region": "Asia"},
                          capacity=85, resilience_factor=0.65, inventory_days=30),
            
            # Assembly & Test
            SupplyChainNode("ASE_TW", "ASE Group", "assembly",
                          {"lat": 22.6, "lon": 120.3, "country": "Taiwan", "region": "Asia"},
                          capacity=90, resilience_factor=0.6, inventory_days=20),
            SupplyChainNode("AMKOR_KR", "Amkor", "assembly",
                          {"lat": 37.4, "lon": 126.7, "country": "South Korea", "region": "Asia"},
                          capacity=80, resilience_factor=0.65, inventory_days=20),
            
            # Key customers/integrators
            SupplyChainNode("APPLE_US", "Apple", "customer",
                          {"lat": 37.3, "lon": -122.0, "country": "USA", "region": "Americas"},
                          capacity=100, resilience_factor=0.8, inventory_days=15),
            SupplyChainNode("NVIDIA_US", "NVIDIA", "customer",
                          {"lat": 37.3, "lon": -121.9, "country": "USA", "region": "Americas"},
                          capacity=95, resilience_factor=0.75, inventory_days=20),
            
            # Logistics hubs
            SupplyChainNode("SINGAPORE_PORT", "Singapore Port", "logistics",
                          {"lat": 1.3, "lon": 103.8, "country": "Singapore", "region": "Asia"},
                          capacity=100, resilience_factor=0.85, inventory_days=5),
            SupplyChainNode("SHANGHAI_PORT", "Shanghai Port", "logistics",
                          {"lat": 31.2, "lon": 121.5, "country": "China", "region": "Asia"},
                          capacity=100, resilience_factor=0.7, inventory_days=5),
        ]
        
        # Add nodes to network
        for node in nodes_data:
            self.nodes[node.id] = node
            self.network.add_node(node.id, **node.__dict__)
        
        # Define critical supply chain edges
        edges_data = [
            # Material flow to foundries
            SupplyChainEdge("SHIN_ETSU_JP", "TSMC_TW", 100, 75, 14, 0.8, "sea"),
            SupplyChainEdge("SUMCO_JP", "SAMSUNG_KR", 90, 70, 10, 0.7, "sea"),
            SupplyChainEdge("SHIN_ETSU_JP", "INTEL_US", 80, 60, 21, 0.6, "sea"),
            
            # Equipment to foundries
            SupplyChainEdge("ASML_NL", "TSMC_TW", 100, 90, 30, 0.95, "air"),
            SupplyChainEdge("ASML_NL", "SAMSUNG_KR", 80, 70, 30, 0.9, "air"),
            SupplyChainEdge("AMAT_US", "INTEL_US", 90, 80, 7, 0.8, "road"),
            
            # Foundry to assembly
            SupplyChainEdge("TSMC_TW", "ASE_TW", 100, 85, 2, 0.9, "road"),
            SupplyChainEdge("SAMSUNG_KR", "AMKOR_KR", 90, 75, 1, 0.85, "road"),
            
            # Assembly to customers
            SupplyChainEdge("ASE_TW", "APPLE_US", 100, 90, 7, 0.95, "air"),
            SupplyChainEdge("TSMC_TW", "NVIDIA_US", 95, 85, 7, 0.9, "air"),
            
            # Logistics connections
            SupplyChainEdge("SINGAPORE_PORT", "TSMC_TW", 100, 60, 3, 0.5, "sea"),
            SupplyChainEdge("SHANGHAI_PORT", "SAMSUNG_KR", 100, 55, 2, 0.4, "sea"),
        ]
        
        # Add edges to network
        for edge in edges_data:
            self.edges[(edge.source, edge.target)] = edge
            self.network.add_edge(edge.source, edge.target, **edge.__dict__)
        
        return self.network
    
    def simulate_scenario(self, scenario: Scenario, verbose: bool = True) -> Dict[str, Any]:
        """
        Simulate a disruption scenario and track risk propagation
        """
        if verbose:
            print(f"🎯 Simulating Scenario: {scenario.name}")
            print(f"   Type: {scenario.disruption_type.value}")
            print(f"   Epicenter: {scenario.epicenter_nodes}")
            print(f"   Duration: {scenario.duration_days} days")
            print("-" * 50)
        
        # Initialize simulation state
        timesteps = self.config['simulation_timesteps']
        risk_evolution = {node: [] for node in self.nodes}
        cascade_events = []
        economic_impact = []
        
        # Set initial disruption
        for node_id in scenario.epicenter_nodes:
            if node_id in self.nodes:
                self.current_risks[node_id] = scenario.initial_impact
        
        # Run simulation
        for t in range(timesteps):
            daily_risks = {}
            daily_cascades = []
            
            # Propagate risk through network
            for node_id in self.network.nodes():
                new_risk = self._calculate_node_risk(
                    node_id, scenario, t, timesteps
                )
                daily_risks[node_id] = new_risk
                risk_evolution[node_id].append(new_risk)
                
                # Check for cascade events
                if new_risk > self.config['risk_threshold']:
                    if self.current_risks.get(node_id, 0) <= self.config['risk_threshold']:
                        daily_cascades.append({
                            'node': node_id,
                            'timestep': t,
                            'risk': new_risk,
                            'type': self.nodes[node_id].type
                        })
            
            # Update current risks
            self.current_risks = daily_risks
            cascade_events.extend(daily_cascades)
            
            # Calculate economic impact
            daily_impact = self._calculate_economic_impact(daily_risks)
            economic_impact.append(daily_impact)
            
            # Apply recovery
            self._apply_recovery(scenario.recovery_rate, t, timesteps)
        
        # Analyze results
        results = self._analyze_simulation_results(
            scenario, risk_evolution, cascade_events, economic_impact
        )
        
        if verbose:
            self._print_simulation_summary(results)
        
        self.scenario_results[scenario.id] = results
        return results
    
    def _calculate_node_risk(self, node_id: str, scenario: Scenario, 
                            t: int, max_t: int) -> float:
        """Calculate risk for a node at time t using selected propagation model"""
        
        node = self.nodes[node_id]
        base_risk = self.current_risks.get(node_id, 0)
        
        # Get risk from connected nodes
        predecessor_risk = 0
        for pred in self.network.predecessors(node_id):
            edge = self.edges.get((pred, node_id))
            if edge:
                pred_risk = self.current_risks.get(pred, 0)
                # Risk transmission depends on dependency and edge capacity
                transmission = pred_risk * edge.dependency_strength * (edge.current_flow / edge.capacity)
                predecessor_risk = max(predecessor_risk, transmission)
        
        # Apply propagation model
        if scenario.propagation_model == PropagationModel.LINEAR:
            propagated_risk = base_risk + (predecessor_risk * scenario.propagation_speed)
        
        elif scenario.propagation_model == PropagationModel.EXPONENTIAL:
            growth_rate = scenario.propagation_speed * 0.1
            propagated_risk = base_risk + predecessor_risk * (1 + growth_rate) ** (t / 10)
        
        elif scenario.propagation_model == PropagationModel.SIGMOID:
            # S-curve propagation
            midpoint = max_t / 2
            steepness = 0.2
            sigmoid = 1 / (1 + np.exp(-steepness * (t - midpoint)))
            propagated_risk = base_risk + predecessor_risk * sigmoid
        
        elif scenario.propagation_model == PropagationModel.THRESHOLD:
            # Risk jumps when threshold exceeded
            threshold = 50
            if predecessor_risk > threshold:
                propagated_risk = min(base_risk + predecessor_risk * 0.8, 100)
            else:
                propagated_risk = base_risk * 0.95  # Natural decay
        
        else:  # STOCHASTIC
            # Add random component
            noise = np.random.normal(0, 5)
            propagated_risk = base_risk + (predecessor_risk * scenario.propagation_speed) + noise
        
        # Apply node resilience
        mitigated_risk = propagated_risk * (1 - node.resilience_factor * 0.3)
        
        # Apply inventory buffer effect
        inventory_buffer = min(node.inventory_days / 30, 1)
        final_risk = mitigated_risk * (1 - inventory_buffer * 0.2)
        
        return min(max(final_risk, 0), 100)
    
    def _apply_recovery(self, recovery_rate: float, t: int, max_t: int):
        """Apply recovery to all nodes"""
        recovery_factor = recovery_rate * (t / max_t) * 0.1
        for node_id in self.current_risks:
            self.current_risks[node_id] *= (1 - recovery_factor)
    
    def _calculate_economic_impact(self, risks: Dict[str, float]) -> float:
        """Calculate economic impact based on risk levels"""
        total_impact = 0
        for node_id, risk in risks.items():
            node = self.nodes[node_id]
            # Impact = capacity * risk * criticality
            criticality = 1.5 if node.type == "foundry" else 1.0
            impact = node.capacity * (risk / 100) * criticality * 1000000  # in dollars
            total_impact += impact
        return total_impact
    
    def _analyze_simulation_results(self, scenario: Scenario, 
                                   risk_evolution: Dict[str, List[float]],
                                   cascade_events: List[Dict],
                                   economic_impact: List[float]) -> Dict[str, Any]:
        """Analyze and summarize simulation results"""
        
        # Calculate key metrics
        peak_risks = {node: max(risks) for node, risks in risk_evolution.items()}
        avg_risks = {node: np.mean(risks) for node, risks in risk_evolution.items()}
        
        # Find most affected nodes
        most_affected = sorted(peak_risks.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate cascade metrics
        cascade_count = len(cascade_events)
        cascade_timeline = {}
        for event in cascade_events:
            t = event['timestep']
            if t not in cascade_timeline:
                cascade_timeline[t] = []
            cascade_timeline[t].append(event['node'])
        
        # Economic metrics
        total_impact = sum(economic_impact)
        peak_impact = max(economic_impact)
        recovery_time = next((i for i, impact in enumerate(economic_impact) 
                             if impact < peak_impact * 0.1), len(economic_impact))
        
        # Network metrics
        affected_nodes = sum(1 for peak in peak_risks.values() if peak > 30)
        network_resilience = 1 - (affected_nodes / len(self.nodes))
        
        # Calculate supply chain bottlenecks
        bottlenecks = self._identify_bottlenecks(risk_evolution)
        
        return {
            'scenario': scenario.__dict__,
            'peak_risks': peak_risks,
            'average_risks': avg_risks,
            'most_affected_nodes': most_affected,
            'cascade_events': cascade_events,
            'cascade_count': cascade_count,
            'cascade_timeline': cascade_timeline,
            'total_economic_impact': total_impact,
            'peak_economic_impact': peak_impact,
            'recovery_time_steps': recovery_time,
            'network_resilience': network_resilience,
            'affected_nodes_count': affected_nodes,
            'risk_evolution': risk_evolution,
            'economic_impact_timeline': economic_impact,
            'bottlenecks': bottlenecks,
            'simulation_timestamp': datetime.now().isoformat()
        }
    
    def _identify_bottlenecks(self, risk_evolution: Dict[str, List[float]]) -> List[Dict]:
        """Identify supply chain bottlenecks based on risk propagation patterns"""
        bottlenecks = []
        
        for node_id in self.network.nodes():
            # Calculate betweenness centrality
            paths_through = nx.betweenness_centrality(self.network).get(node_id, 0)
            
            # Calculate risk amplification
            successors = list(self.network.successors(node_id))
            if successors:
                node_peak = max(risk_evolution[node_id])
                successor_peaks = [max(risk_evolution[s]) for s in successors]
                amplification = np.mean(successor_peaks) / (node_peak + 1)
                
                if paths_through > 0.1 or amplification > 1.5:
                    bottlenecks.append({
                        'node': node_id,
                        'centrality': paths_through,
                        'risk_amplification': amplification,
                        'downstream_nodes': len(successors),
                        'criticality_score': paths_through * amplification
                    })
        
        return sorted(bottlenecks, key=lambda x: x['criticality_score'], reverse=True)[:5]
    
    def _print_simulation_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of simulation results"""
        print("\n📊 SIMULATION RESULTS")
        print("=" * 50)
        
        print(f"\n🎯 Network Impact:")
        print(f"   Affected Nodes: {results['affected_nodes_count']}/{len(self.nodes)}")
        print(f"   Network Resilience: {results['network_resilience']:.2%}")
        print(f"   Cascade Events: {results['cascade_count']}")
        
        print(f"\n💰 Economic Impact:")
        print(f"   Total Impact: ${results['total_economic_impact']/1e9:.2f}B")
        print(f"   Peak Impact: ${results['peak_economic_impact']/1e9:.2f}B")
        print(f"   Recovery Time: {results['recovery_time_steps']} days")
        
        print(f"\n🔥 Most Affected Nodes:")
        for node, risk in results['most_affected_nodes'][:3]:
            node_obj = self.nodes[node]
            print(f"   • {node_obj.name}: {risk:.1f}% peak risk")
        
        print(f"\n🔗 Critical Bottlenecks:")
        for bottleneck in results['bottlenecks'][:3]:
            node_obj = self.nodes[bottleneck['node']]
            print(f"   • {node_obj.name}: {bottleneck['criticality_score']:.2f} criticality")
    
    def run_monte_carlo_analysis(self, base_scenario: Scenario, 
                                runs: int = 100) -> Dict[str, Any]:
        """Run Monte Carlo simulation for uncertainty analysis"""
        
        print(f"\n🎲 Running Monte Carlo Analysis ({runs} simulations)...")
        
        all_results = []
        
        for i in range(runs):
            # Vary scenario parameters
            varied_scenario = Scenario(
                id=f"{base_scenario.id}_mc_{i}",
                name=f"{base_scenario.name} (MC {i})",
                description=base_scenario.description,
                disruption_type=base_scenario.disruption_type,
                epicenter_nodes=base_scenario.epicenter_nodes,
                initial_impact=base_scenario.initial_impact * np.random.uniform(0.8, 1.2),
                duration_days=int(base_scenario.duration_days * np.random.uniform(0.9, 1.1)),
                propagation_model=base_scenario.propagation_model,
                propagation_speed=base_scenario.propagation_speed * np.random.uniform(0.7, 1.3),
                recovery_rate=base_scenario.recovery_rate * np.random.uniform(0.8, 1.2)
            )
            
            # Run simulation
            result = self.simulate_scenario(varied_scenario, verbose=False)
            all_results.append(result)
        
        # Analyze Monte Carlo results
        economic_impacts = [r['total_economic_impact'] for r in all_results]
        recovery_times = [r['recovery_time_steps'] for r in all_results]
        cascade_counts = [r['cascade_count'] for r in all_results]
        
        analysis = {
            'runs': runs,
            'economic_impact': {
                'mean': np.mean(economic_impacts),
                'std': np.std(economic_impacts),
                'min': np.min(economic_impacts),
                'max': np.max(economic_impacts),
                'percentiles': {
                    '5%': np.percentile(economic_impacts, 5),
                    '25%': np.percentile(economic_impacts, 25),
                    '50%': np.percentile(economic_impacts, 50),
                    '75%': np.percentile(economic_impacts, 75),
                    '95%': np.percentile(economic_impacts, 95)
                }
            },
            'recovery_time': {
                'mean': np.mean(recovery_times),
                'std': np.std(recovery_times),
                'min': np.min(recovery_times),
                'max': np.max(recovery_times)
            },
            'cascade_events': {
                'mean': np.mean(cascade_counts),
                'std': np.std(cascade_counts),
                'min': np.min(cascade_counts),
                'max': np.max(cascade_counts)
            },
            'confidence_intervals': {
                '90%': (np.percentile(economic_impacts, 5), np.percentile(economic_impacts, 95)),
                '95%': (np.percentile(economic_impacts, 2.5), np.percentile(economic_impacts, 97.5))
            }
        }
        
        print(f"\n📈 Monte Carlo Results:")
        print(f"   Economic Impact: ${analysis['economic_impact']['mean']/1e9:.2f}B ± ${analysis['economic_impact']['std']/1e9:.2f}B")
        print(f"   90% Confidence: ${analysis['confidence_intervals']['90%'][0]/1e9:.2f}B - ${analysis['confidence_intervals']['90%'][1]/1e9:.2f}B")
        print(f"   Recovery Time: {analysis['recovery_time']['mean']:.1f} ± {analysis['recovery_time']['std']:.1f} days")
        
        return analysis
    
    def export_results_for_dashboard(self, scenario_id: str) -> Dict[str, Any]:
        """Export results in format ready for dashboard visualization"""
        
        if scenario_id not in self.scenario_results:
            raise ValueError(f"Scenario {scenario_id} not found in results")
        
        results = self.scenario_results[scenario_id]
        
        # Format for dashboard
        dashboard_data = {
            'metadata': {
                'scenario_name': results['scenario']['name'],
                'disruption_type': results['scenario']['disruption_type'],
                'simulation_time': results['simulation_timestamp'],
                'network_nodes': len(self.nodes),
                'network_edges': len(self.edges)
            },
            'summary_metrics': {
                'total_impact': f"${results['total_economic_impact']/1e9:.2f}B",
                'recovery_time': f"{results['recovery_time_steps']} days",
                'affected_nodes': results['affected_nodes_count'],
                'cascade_events': results['cascade_count'],
                'network_resilience': f"{results['network_resilience']:.1%}"
            },
            'time_series': {
                'timestamps': list(range(len(results['economic_impact_timeline']))),
                'economic_impact': results['economic_impact_timeline'],
                'risk_evolution': {
                    node: risks for node, risks in results['risk_evolution'].items()
                    if max(risks) > 30  # Only include significantly affected nodes
                }
            },
            'network_view': {
                'nodes': [
                    {
                        'id': node_id,
                        'label': self.nodes[node_id].name,
                        'type': self.nodes[node_id].type,
                        'peak_risk': results['peak_risks'][node_id],
                        'lat': self.nodes[node_id].location['lat'],
                        'lon': self.nodes[node_id].location['lon'],
                        'size': max(10, results['peak_risks'][node_id] / 2)
                    }
                    for node_id in self.nodes
                ],
                'edges': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'weight': self.edges[edge].dependency_strength
                    }
                    for edge in self.edges
                ]
            },
            'bottlenecks': results['bottlenecks'],
            'cascade_timeline': results['cascade_timeline']
        }
        
        return dashboard_data


# Example usage and test scenarios
if __name__ == "__main__":
    # Initialize simulator
    simulator = SupplyChainNetworkSimulator()
    
    # Build network
    network = simulator.build_semiconductor_network()
    print(f"✅ Built network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
    
    # Define test scenarios
    scenarios = [
        Scenario(
            id="taiwan_earthquake",
            name="Taiwan Earthquake Scenario",
            description="Major earthquake affecting Taiwan's semiconductor infrastructure",
            disruption_type=DisruptionType.NATURAL_DISASTER,
            epicenter_nodes=["TSMC_TW", "ASE_TW"],
            initial_impact=85,
            duration_days=30,
            propagation_model=PropagationModel.EXPONENTIAL,
            propagation_speed=0.7,
            recovery_rate=0.3
        ),
        Scenario(
            id="china_tensions",
            name="China-Taiwan Geopolitical Tensions",
            description="Escalation of geopolitical tensions affecting shipping and trade",
            disruption_type=DisruptionType.GEOPOLITICAL,
            epicenter_nodes=["SHANGHAI_PORT", "SINGAPORE_PORT"],
            initial_impact=70,
            duration_days=60,
            propagation_model=PropagationModel.SIGMOID,
            propagation_speed=0.5,
            recovery_rate=0.2
        ),
        Scenario(
            id="cyber_attack",
            name="Supply Chain Cyber Attack",
            description="Coordinated cyber attack on semiconductor equipment manufacturers",
            disruption_type=DisruptionType.CYBER_ATTACK,
            epicenter_nodes=["ASML_NL", "AMAT_US"],
            initial_impact=60,
            duration_days=14,
            propagation_model=PropagationModel.THRESHOLD,
            propagation_speed=0.9,
            recovery_rate=0.5
        )
    ]
    
    # Run simulations
    for scenario in scenarios:
        print(f"\n{'='*60}")
        results = simulator.simulate_scenario(scenario)
        
        # Run Monte Carlo analysis for the first scenario
        if scenario.id == "taiwan_earthquake":
            mc_results = simulator.run_monte_carlo_analysis(scenario, runs=100)
    
    # Export results for dashboard
    print(f"\n{'='*60}")
    print("📤 Exporting results for dashboard...")
    dashboard_data = simulator.export_results_for_dashboard("taiwan_earthquake")
    
    # Save to JSON
    with open('data/processed/scenario_simulation_results.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print("✅ Results exported to data/processed/scenario_simulation_results.json")
    print(f"\n🎯 Simulation Complete!")
    print(f"   Total scenarios simulated: {len(scenarios)}")
    print(f"   Network complexity: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    print(f"   Ready for dashboard visualization!")