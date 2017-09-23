/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 150;

	// normal distributions for sensor noise
	normal_distribution<double> norm_x(x, std[0]);
	normal_distribution<double> norm_y(y, std[1]);
	normal_distribution<double> norm_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {

		particles.push_back(Particle { i, norm_x(generator), norm_y(generator),
				norm_theta(generator), 1.0 });

		weights.push_back(1.0);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
		double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// normal distributions for sensor noise
	normal_distribution<double> norm_x(0, std_pos[0]);
	normal_distribution<double> norm_y(0, std_pos[1]);
	normal_distribution<double> norm_theta(0, std_pos[2]);

	for (Particle& particle : particles) {

		// predict new state

		if (fabs(yaw_rate) < 0.00001) {
			// prevent 0 division
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);

		} else {

			double dvdyaw = velocity / yaw_rate;
			double yawdt = yaw_rate * delta_t;

			particle.x += dvdyaw
					* (sin(particle.theta + yaw_rate * delta_t)
							- sin(particle.theta));
			particle.y += dvdyaw
					* (cos(particle.theta) - cos(particle.theta + yawdt));
			particle.theta += yawdt;
		}

		// add some noise
		particle.x += norm_x(generator);
		particle.y += norm_y(generator);
		particle.theta += norm_theta(generator);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
		std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (LandmarkObs& obs : observations) {

		double min_dist = numeric_limits<double>::max();
		int map_id = -1;

		for (const LandmarkObs& pred : predicted) {
			double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);

			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				map_id = pred.id;
			}
		}

		obs.id = map_id;
	}
}

vector<LandmarkObs> ParticleFilter::filterLandmarks(const Particle& particle,
		const Map &map_landmarks, double sensor_range) {

	double x_p = particle.x;
	double y_p = particle.y;

	vector<LandmarkObs> landmarks_to_keep;
	for (const Map::single_landmark_s& landmark : map_landmarks.landmark_list) {
		int id_i = landmark.id_i;
		float x_f = landmark.x_f;
		float y_f = landmark.y_f;

		if (dist(x_f, y_f, x_p, y_p) <= sensor_range) {
			landmarks_to_keep.push_back(LandmarkObs { id_i, x_f, y_f });
		}
	}
	return landmarks_to_keep;
}

// transform observations
vector<LandmarkObs> ParticleFilter::transformObservations(
		const Particle& particle,
		const std::vector<LandmarkObs> &observations) {

	double x_p = particle.x;
	double y_p = particle.y;
	double theta_p = particle.theta;

	vector<LandmarkObs> transformed_obs;
	for (const LandmarkObs& obs : observations) {
		double t_x = x_p + cos(theta_p) * obs.x - sin(theta_p) * obs.y;
		double t_y = y_p + sin(theta_p) * obs.x + cos(theta_p) * obs.y;

		transformed_obs.push_back(LandmarkObs { obs.id, t_x, t_y });

	}

	return transformed_obs;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations,
		const Map &map_landmarks) {
// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
//   according to the MAP'S coordinate system. You will need to transform between the two systems.
//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
//   The following is a good resource for the theory:
//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
//   and the following is a good resource for the actual equation to implement (look at equation
//   3.33
//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; ++i) {

		Particle& particle = particles[i];

		vector<LandmarkObs> predictions = filterLandmarks(particle,
				map_landmarks, sensor_range);

		vector<LandmarkObs> transformed_obs = transformObservations(particle,
				observations);

		dataAssociation(predictions, transformed_obs);

		// reset weight
		particle.weight = 1.0;

		for (const LandmarkObs& obs : transformed_obs) {
			double x_pred, y_pred;

			// get coordinates of the prediction associated with the current observation
			for (const LandmarkObs& prediction : predictions) {
				if (prediction.id == obs.id) {
					x_pred = prediction.x;
					y_pred = prediction.y;
					break;
				}
			}

			// importance weight for this observation with multivariate Gaussian distribution
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double w = 1.0 / (2.0 * M_PI * std_x * std_y)
					* exp(
							-pow(x_pred - obs.x, 2) / (2 * std_x * std_x)
									- pow(y_pred - obs.y, 2)
											/ (2 * std_y * std_y));

			// accumulate weight
			particle.weight *= w;
		}

		weights[i] = particle.weight;
	}
}

void ParticleFilter::resample() {
// TODO: Resample particles with replacement with probability proportional to their weight.
// NOTE: You may find std::discrete_distribution helpful here.
//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::discrete_distribution<int> discrete_dist(weights.begin(), weights.end());
	vector<Particle> resampled;

	for (int i = 0; i < num_particles; ++i) {
		int index = discrete_dist(generator);
		resampled.push_back(particles[index]);
	}

	particles = resampled;

}

Particle ParticleFilter::SetAssociations(Particle particle,
		std::vector<int> associations, std::vector<double> sense_x,
		std::vector<double> sense_y) {
//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
// associations: The landmark id that goes along with each listed association
// sense_x: the associations x mapping already converted to world coordinates
// sense_y: the associations y mapping already converted to world coordinates

//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
