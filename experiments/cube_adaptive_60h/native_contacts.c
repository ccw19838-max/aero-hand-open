#include <mujoco/mujoco.h>
#include <string.h>

/* Read-only equivalents of the frozen Python contact readers. */
int cube_forbidden(const mjModel* model, const mjData* data, int cube,
                   const unsigned char* forbidden, double threshold) {
  mjtNum force[6];
  for (int i = 0; i < data->ncon; ++i) {
    const mjContact* contact = data->contact + i;
    int other = contact->geom[0] == cube ? contact->geom[1] :
                contact->geom[1] == cube ? contact->geom[0] : -1;
    if (other < 0 || !forbidden[other]) continue;
    mj_contactForce(model, data, i, force);
    if (force[0] >= threshold) return 1;
  }
  return 0;
}

void cube_contacts(const mjModel* model, const mjData* data, int cube,
                   const int* fingers, const unsigned char* palm,
                   const unsigned char* forbidden, double threshold,
                   double* output) {
  memset(output, 0, 8 * sizeof(double));
  mjtNum force[6];
  for (int i = 0; i < data->ncon; ++i) {
    const mjContact* contact = data->contact + i;
    int other = contact->geom[0] == cube ? contact->geom[1] :
                contact->geom[1] == cube ? contact->geom[0] : -1;
    if (other < 0) continue;
    mj_contactForce(model, data, i, force);
    double normal = force[0] > 0.0 ? force[0] : 0.0;
    int finger = fingers[other];
    if (finger >= 0 && finger < 5) output[finger] += normal;
    if (palm[other]) output[5] += normal;
    double depth = -contact->dist;
    if (depth > output[6]) output[6] = depth;
    if (forbidden[other] && force[0] >= threshold) output[7] = 1.0;
  }
}
