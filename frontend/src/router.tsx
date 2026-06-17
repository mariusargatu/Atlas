import { Chat } from "@/routes/Chat";
import { Login } from "@/routes/Login";
import { Root } from "@/routes/Root";
import { createRootRoute, createRoute, createRouter } from "@tanstack/react-router";

const rootRoute = createRootRoute({ component: Root });
const loginRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: Login });
const chatRoute = createRoute({ getParentRoute: () => rootRoute, path: "/chat", component: Chat });

const routeTree = rootRoute.addChildren([loginRoute, chatRoute]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
