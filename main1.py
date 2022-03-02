from __future__ import print_function
import torch.optim as optim
from torch.utils.data import TensorDataset
from flows2 import *
from train_utils import *
from plot_utils import *
from data_utils import *
import argparse

parser = argparse.ArgumentParser(description='modified_attention')

parser.add_argument(
    '--data-name',  # learn
    type=str,
    default='Sshape',#
    help='name of the letter in LASA dataset')

args = parser.parse_args()

data_name = args.data_name
test_learner_model = True  # to plot the rollouts and vector fields
load_learner_model = False  # to load a saved model
plot_resolution = 0.01  # plotting resolution (only use for testing)

eps = 1e-12
cuda = False
seed = None
weight_regularizer = 0.01
epochs = 1000
loss_clip = 1e3
clip_gradient = True
clip_value_grad = 0.1
log_freq = 10
plot_freq = 120
stopping_thresh = 2500
import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "2"
device =torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")

minibatch_mode = False  # True uses the batch_size arg below
learning_rate = 0.01
batch_size = 32  # size of minibatch

if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

# ---------------------------------------------------------------
print('Loading dataset...')

dataset = LASA(data_name=data_name)

goal = dataset.goal
idx = dataset.idx

x_train = dataset.x
xd_train = dataset.xd

scaling = torch.from_numpy(dataset.scaling).float()
translation = torch.from_numpy(dataset.translation).float()

normalize_ = lambda x: x * scaling + translation
denormalize_ = lambda x: (x - translation) / scaling

n_dims = dataset.n_dims
n_pts = dataset.n_pts

dt = dataset.dt


dataset_list = []
time_list = []
expert_traj_list = []
s0_list = []
t_final_list = []

for n in range(len(idx) - 1):
    x_traj_tensor = torch.from_numpy(x_train[idx[n]:idx[n + 1]])
    x_traj_tensor[:, 0] = x_traj_tensor[:, 0] - goal[0][0]
    x_traj_tensor[:, 1] = x_traj_tensor[:, 1] - goal[0][1]

    xd_traj_tensor = torch.from_numpy(xd_train[idx[n]:idx[n + 1]])
    s0_list.append(x_traj_tensor[0].numpy())
    traj_dataset = torch.utils.data.TensorDataset(x_traj_tensor, xd_traj_tensor)
    expert_traj_list.append(x_traj_tensor)
    dataset_list.append(traj_dataset)
    t_final = dt * (x_traj_tensor.shape[0] - 1)
    t_final_list.append(t_final)
    t_eval = np.arange(0., t_final + dt, dt)
    time_list.append(t_eval)

n_experts = len(dataset_list)

x_train_tensor = torch.from_numpy(x_train).to(device)

xd_train_tensor = torch.from_numpy(xd_train).to(device)

if not minibatch_mode:
    batch_size = xd_train.shape[0]

#  ------------------------------------------
# finding the data range

xmin = np.min(x_train[:, 0])
xmax = np.max(x_train[:, 0])
ymin = np.min(x_train[:, 1])
ymax = np.max(x_train[:, 1])

x_lim = [[xmin-0.1 , xmax+0.1 ], [ymin-0.2 , ymax+0.2 ]]

# --------------------------------------------------------------------------------

learner_model = TransAm().to(device)

if not load_learner_model:
    print('Training model ...')
    # Training learner
    optimizer = optim.AdamW(learner_model.parameters(), lr=learning_rate, weight_decay=weight_regularizer)
    criterion = nn.SmoothL1Loss()

    loss_fn = criterion

    dataset = TensorDataset(x_train_tensor, xd_train_tensor)
    best_model, train_loss = \
        train(learner_model, loss_fn, optimizer, dataset, epochs, batch_size=batch_size, stop_threshold=stopping_thresh)

    print(
        'Training loss: {:.4f}'.
            format(train_loss))

    try:
        os.makedirs('models')
    except OSError:
        pass

    learner_model = best_model.to(device)
    torch.save(learner_model.state_dict(), os.path.join('models', '{}.pt'.format(data_name)))

else:
    print('Loading model ...')
    # Loading learner
    learner_model.load_state_dict(torch.load(os.path.join('models', '{}.pt'.format(data_name))))

# ---------------------------------------------------------
# Plotting best results

if test_learner_model:
    print('Plotting rollouts and vector fields. This may take a few moments ...')
    learner_traj_list = []

    # rollout trajectories
    for n in range(n_experts):
        s0 = s0_list[n]
        s0 = torch.Tensor(s0.reshape(-1)).to(device)
        # s0 = torch.Tensor((xmin-0.6,ymin-0.2)).to(device)

        t_final = t_final_list[n]
        learner_traj = generate_trajectories(learner_model, s0, order=1, return_label=False, t_step=dt,
                                             t_final=1*t_final)
        # print(learner_traj)

        learner_traj = learner_traj[0]

        learner_traj_list.append(learner_traj)

    # visualize vector field and potentials
    taskmap_net = learner_model
    potential_fcn = lambda x: torch.norm(taskmap_net(x)[0], dim=2)

    x1_test = np.arange(x_lim[0][0], x_lim[0][1], plot_resolution)
    x2_test = np.arange(x_lim[1][0], x_lim[1][1], plot_resolution)
    X1, X2 = np.meshgrid(x1_test, x2_test)
    x_test = np.concatenate((X1.flatten().reshape(-1, 1), X2.flatten().reshape(-1, 1)), 1)
    x_test_tensor = torch.from_numpy(x_test).float().to(device)

    z_test_tensor = potential_fcn(x_test_tensor.unsqueeze(0))
    z_test = z_test_tensor.detach().cpu().numpy()
    max_z = np.max(z_test)
    min_z = np.min(z_test)
    z_test = (z_test - min_z) / (max_z - min_z)

    Z = z_test.reshape(X1.shape[0], X1.shape[1])

    fig1 = plt.figure()
    ax1 = plt.gca()
    ax1.set_xlim(x_lim[0])
    ax1.set_ylim(x_lim[1])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    ax1.imshow(Z, extent=[x_lim[0][0], x_lim[0][1], x_lim[1][0], x_lim[1][1]], origin='lower', cmap='viridis')
    # ax1.axis(aspect='image')
    x_lim = torch.Tensor(x_lim).to(device)

    visualize_vel(learner_model, x_lim=x_lim, delta=0.011, cmap=None, color='#f2e68f')

    expert_traj_list = [traj.numpy() for traj in expert_traj_list]
    learner_traj_list = [traj.numpy() for traj in learner_traj_list]
    sum = 0

    for n in range(n_experts):
        expert_traj = expert_traj_list[n]
        learner_traj = learner_traj_list[n]

        ax1.plot(expert_traj[:, 0], expert_traj[:, 1], 'w', linewidth=4, linestyle=':')
        ax1.plot(learner_traj[:, 0], learner_traj[:, 1], 'r', linewidth=3)
        ax1.plot(expert_traj[-1, 0], expert_traj[-1, 1], 'o', linewidth=10, markersize=8, markeredgecolor='black')

    try:
        os.makedirs('plots')
    except OSError:
        pass

    fig1.savefig(os.path.join('plots', '{}_vector_field.pdf'.format(data_name)), dpi=300)

    plt.show()





