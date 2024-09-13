#this project aims to create a pipeline of how federated learning works
import hydra
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
import flwr as fl
from fit_strategy import get_on_fit_config, get_evaluate_fn
import pickle

@hydra.main(config_path = "conf", config_name = "base", version_base = None)
def main(cfg : DictConfig) :
    # 1. Parse config and get expirement output directory
    print("main initiated")
    print(OmegaConf.to_yaml(cfg))
    print(cfg)

    # 2. Prepare the dataset
    trainloader, validationloader, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    # 3. Define the clients
    client_fn = generate_client_fn(trainloader, validationloader, cfg.num_classes)

    # 4. Define the strategy that will be used
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.0001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.0001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader))

    # 5. Start the simulation
    history  = fl.simulation.start_simulation(
        client_fn = client_fn,
        num_clients = cfg.num_clients,
        config = fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy = strategy,
        #with this we can set the number of how many clients can run parallel in the gpu during the simulation (e.g. num_gpus : 0.5 means that 2 clients are able to use it each taking half he space)
        #client_resources = {'num_cpus':2, 'num_gpus':1.0} 
    )

    # 6. Save the results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path)/'results.pkl'

    results = {'history' : history}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()