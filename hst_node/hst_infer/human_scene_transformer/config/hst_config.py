# DatasetParam
from hst_infer.human_scene_transformer.jrdb.dataset_params import JRDBDatasetParams
from hst_infer.human_scene_transformer.model.model_params import ModelParams

from hst_infer.human_scene_transformer import is_hidden_generators
from hst_infer.human_scene_transformer.model import agent_feature_encoder
from hst_infer.human_scene_transformer.model import head
from hst_infer.human_scene_transformer.model import scene_encoder as _

from hst_infer.node_config import WINDOW_LENGTH, HISTORY_LENGTH

TEST_SCENES = \
    ['clark-center-2019-02-28_1',
     'forbes-cafe-2019-01-22_0',
     'gates-basement-elevators-2019-01-17_1',
     'huang-2-2019-01-25_0',
     'jordan-hall-2019-04-22_0',
     'nvidia-aud-2019-04-18_0',
     'packard-poster-session-2019-03-20_2',
     'svl-meeting-gates-2-2019-04-08_1',
     'tressider-2019-04-26_2',
     'discovery-walk-2019-02-28_1_test',
     'gates-basement-elevators-2019-01-17_0_test',
     'hewlett-class-2019-01-23_0_test',
     'huang-intersection-2019-01-22_0_test',
     'meyer-green-2019-03-16_1_test',
     'nvidia-aud-2019-04-18_2_test',
     'serra-street-2019-01-30_0_test',
     'tressider-2019-03-16_2_test',
     'tressider-2019-04-26_3_test']

TRAIN_SCENES = \
    ['bytes-cafe-2019-02-07_0',
     'clark-center-2019-02-28_0',
     'clark-center-intersection-2019-02-28_0',
     'cubberly-auditorium-2019-04-22_0',
     'gates-159-group-meeting-2019-04-03_0',
     'gates-ai-lab-2019-02-08_0',
     'gates-to-clark-2019-02-28_1',
     'hewlett-packard-intersection-2019-01-24_0',
     'huang-basement-2019-01-25_0',
     'huang-lane-2019-02-12_0',
     'memorial-court-2019-03-16_0',
     'meyer-green-2019-03-16_0',
     'packard-poster-session-2019-03-20_0',
     'packard-poster-session-2019-03-20_1',
     'stlc-111-2019-04-19_0',
     'svl-meeting-gates-2-2019-04-08_0',
     'tressider-2019-03-16_0',
     'tressider-2019-03-16_1',
     'cubberly-auditorium-2019-04-22_1_test',
     'discovery-walk-2019-02-28_0_test',
     'food-trucks-2019-02-12_0_test',
     'gates-ai-lab-2019-04-17_0_test',
     'gates-foyer-2019-01-17_0_test',
     'gates-to-clark-2019-02-28_0_test',
     'hewlett-class-2019-01-23_1_test',
     'huang-2-2019-01-25_1_test',
     'indoor-coupa-cafe-2019-02-06_0_test',
     'lomita-serra-intersection-2019-01-30_0_test',
     'nvidia-aud-2019-01-25_0_test',
     'nvidia-aud-2019-04-18_1_test',
     'outdoor-coupa-cafe-2019-02-06_0_test',
     'quarry-road-2019-02-28_0_test',
     'stlc-111-2019-04-19_1_test',
     'stlc-111-2019-04-19_2_test',
     'tressider-2019-04-26_0_test',
     'tressider-2019-04-26_1_test']


hst_dataset_param = JRDBDatasetParams(
    path=None,
    train_scenes = TRAIN_SCENES,
    train_split = (0.0, 1.0),
    eval_scenes = TEST_SCENES,
    eval_split = (0.0, 1.0),
    features = ['agents/position', 
                'agents/keypoints',
                'robot/position'],
    min_distance_to_robot = 7.0,
    num_agents = 8,
    num_history_steps = int(HISTORY_LENGTH),
    num_pointcloud_points = 512,
    num_steps = int(WINDOW_LENGTH),
    subsample = 6,
    timestep = float(1/3),
)

hst_model_param = ModelParams(
    agents_feature_config= {
        #'agents/keypoints': agent_feature_encoder.AgentKeypointsEncoder,
        'agents/position': agent_feature_encoder.AgentPositionEncoder,
        #'agents/orientation': agent_feature_encoder.Agent2DOrientationEncoder
    },
    agents_position_key= 'agents/position',
    agents_orientation_key= 'agents/orientation',
    attn_architecture=(
        'self-attention',
        'self-attention',
        'multimodality_induction',
        'self-attention',
        'self-attention-mode',
        'self-attention',
        'self-attention-mode',
        ),
    drop_prob= 0.1,
    feature_embedding_size= 128,
    hidden_size= 64,
    is_hidden_generator= is_hidden_generators.BPIsHiddenGenerator,
    ln_eps= 1e-6,
    mask_style= 'has_historic_data',
    num_conv_filters= (32, 32, 64, 64, 128),
    num_heads= 4,
    num_history_steps= int(HISTORY_LENGTH),
    num_modes= 20,
    num_steps= int(WINDOW_LENGTH),
    prediction_head= head.Prediction2DPositionHeadLayer,
    prediction_head_hidden_units= None,

    scene_encoder= None,
    timestep= 0.4,
    transformer_ff_dim= 64,


)































