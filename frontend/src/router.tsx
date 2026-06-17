import { Adjudicate } from "@/routes/Adjudicate";
import { Chat } from "@/routes/Chat";
import { Login } from "@/routes/Login";
import { Root } from "@/routes/Root";
import { createRootRoute, createRoute, createRouter } from "@tanstack/react-router";

const rootRoute = createRootRoute({ component: Root });
const loginRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: Login });
const chatRoute = createRoute({ getParentRoute: () => rootRoute, path: "/chat", component: Chat });
// The HITL adjudication page (SP8 Task 4, label collection half): a standalone internal route, not
// gated behind customer sign in (Login/Chat's own session flow is unrelated to who runs a labeling
// session).
const adjudicateRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/adjudicate",
  component: Adjudicate,
});

const routeTree = rootRoute.addChildren([loginRoute, chatRoute, adjudicateRoute]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
