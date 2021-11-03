import numpy as np
import imageio
from tqdm import tqdm


def scale_image(image, scale=256):
    # determining the length of original image
    w, h = image.shape[:2]

    # xNew and yNew are new width and
    # height of image required after scaling
    xNew = scale
    yNew = scale

    # calculating the scaling factor
    # work for more than 2 pixel
    xScale = xNew / w
    yScale = yNew / h

    # using numpy taking a matrix of xNew
    # width and yNew height with
    # 4 attribute [alpha, B, G, B] values
    new_image = np.zeros([xNew, yNew, 3], dtype=np.uint8)

    for i in range(xNew - 1):
        for j in range(yNew - 1):
            new_image[i, j] = image[int(i / xScale), int(j / yScale)]

    return new_image


def write_video(eval_env, eval_py_env, agent, video_filename='imageio.mp4', num_rounds=10):
    with imageio.get_writer(video_filename, fps=30) as video:
        for _ in tqdm(range(num_rounds)):
            time_step = eval_env.reset()
            video.append_data(scale_image(eval_py_env.render(), scale=512))
            while not time_step.is_last():
                action_step = agent.policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(scale_image(eval_py_env.render(), scale=512))


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for i in range(num_episodes):
        # print(i)
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]
